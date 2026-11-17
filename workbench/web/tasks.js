import { request } from './api.js';
import { createPreparationWorkspace } from './preparation.js';
import { createVersionWorkspace } from './versions.js';
import { createTrainingWorkspace } from './training.js';
import { createTaskNavigation } from './task-navigation.js';
import { createAnnotationWorkspace, stateNames } from './annotations.js';
const $ = (selector) => document.querySelector(selector);
const date = new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });

export function createTaskWorkspace({ notify, onChanged, onContext }) {
  const navigation=createTaskNavigation();
  const training = createTrainingWorkspace({notify});
  const annotations = createAnnotationWorkspace({ notify, onChanged });
  const versions = createVersionWorkspace({notify,onPublished:async()=>{if(parent && current){const result=await request(url(parent.id,current.id));current=result.task;await training.refresh();}}});
  const preparation = createPreparationWorkspace({notify,
    onData: value=>versions.setPreparation(value),
    onImage: async imageId => {
      if (!parent || !current) return;
      const project={...parent},task={...current};
      try {
        const result=await request(`/api/projects/${project.id}/datasets/${task.dataset_id}/images/${imageId}`);
        if (current?.id===task.id && parent?.id===project.id) openRecord(project,task,result.image);
      } catch (error) {notify(error.message);}
    },
    onSaved: async () => {if (parent && current) {const result=await request(url(parent.id,current.id));current=result.task;await versions.open(parent.id,current.id);await training.refresh();}}
  });
  let parent = null, rows = [], current = null, generation = 0, ready = false, lastParent = null;
  let editor = null, busy = false, deletion = null, deleting = false, syncing = false;
  let page = 1, pages = 1, imageEpoch = 0, imageRecords = [], imageTotal = 0, lastImageTask = null;
  const dialog = $('#task-dialog'), form = $('#task-form'), deletionDialog = $('#task-delete-dialog');
  const url = (projectId, id = '') => `/api/projects/${projectId}/tasks${id ? `/${id}` : ''}`;

  function openRecord(project,task,record) {
    // Freeze the filtered navigation before editing so saving a new state cannot skip images.
    let snapshot = null;
    const filters={q:$('#task-image-search').value.trim(),state:$('#task-image-filter').value};
    const navigation={
      prepare: async () => {
        if (snapshot) return;
        const unique=new Map();let next=1,last=1;
        do {
          const params=new URLSearchParams({...filters,page:next,page_size:100});
          const result=await request(`${url(project.id,task.id)}/images?${params}`);
          for (const item of result.images) unique.set(item.id,item);
          last=result.pages;next++;
        } while (next<=last);
        snapshot=[...unique.values()];
        if (!unique.has(record.id)) snapshot.unshift(record);
      },
      position: id => ({index:(snapshot || []).findIndex(item=>item.id===id)+1,total:snapshot?.length || 0}),
      neighbour: async (delta,id) => {const index=snapshot?.findIndex(item=>item.id===id) ?? -1;return index<0 ? null : snapshot[index+delta] || null;},
    };
    annotations.openImage(project.id,task,record,navigation);
  }

  function status(title, text, action, label = '创建任务') {
    const node = $('#tasks-state'); node.replaceChildren(); node.hidden = false;
    const heading = document.createElement('h3'); heading.textContent = title;
    const description = document.createElement('p'); description.textContent = text;
    node.append(heading, description);
    if (action) { const button = document.createElement('button'); button.className = 'button primary'; button.textContent = label; button.addEventListener('click', action); node.append(button); }
  }

  function render() {
    if (!ready) return;
    const query = $('#task-search').value.trim().toLocaleLowerCase();
    const visible = rows.filter((row) => `${row.name}\n${row.description}\n${row.dataset_name}`.toLocaleLowerCase().includes(query));
    if ($('#task-sort').value === 'name') visible.sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'));
    $('#task-grid').replaceChildren(); $('#task-list-count').textContent = rows.length;
    $('#task-filter-label').textContent = query ? `找到 ${visible.length} / ${rows.length} 个任务` : '';
    if (!rows.length) { status('为数据创建第一个任务', '选择任务类型和数据集。多个任务可引用同一批原图，并分别开展后续标注。', () => openEditor()); return; }
    if (!visible.length) { status('没有匹配的任务', '试试其他任务名称、说明或数据集名称。', () => { $('#task-search').value = ''; render(); }, '清除搜索'); return; }
    $('#tasks-state').hidden = true;
    for (const row of visible) {
      const fragment = $('#task-card-template').content.cloneNode(true);
      const link = fragment.querySelector('.task-name-link'); link.textContent = row.name; link.href = `#/projects/${parent.id}/tasks/${row.id}`;
      fragment.querySelector('.task-card-description').textContent = row.description || '暂无任务说明';
      fragment.querySelector('.task-dataset-name').textContent = row.dataset_name;
      fragment.querySelector('.task-image-total').textContent = `已关联 ${row.image_count} 张图片`;
      fragment.querySelector('.task-pending').textContent = row.pending_image_count ? `${row.pending_image_count} 张新图片待同步` : row.class_count ? `${row.class_count} 个类别` : '待设置类别';
      fragment.querySelector('.task-type-badge').textContent=row.task_type==='segment'?'实例分割':'目标检测';
      fragment.querySelector('time').textContent = date.format(new Date(row.updated_at));
      const edit = fragment.querySelector('.edit-task'); edit.setAttribute('aria-label', `编辑任务：${row.name}`); edit.addEventListener('click', () => openEditor(row));
      const remove = fragment.querySelector('.delete-task'); remove.setAttribute('aria-label', `删除任务：${row.name}`); remove.addEventListener('click', () => openDeletion(row));
      $('#task-grid').append(fragment);
    }
  }

  async function loadImages(next = 1) {
    if (!parent || !current) return;
    const task = { ...current }, project = { ...parent }, version = ++imageEpoch;
    page = next; $('#task-image-grid').replaceChildren(); $('#task-image-state').textContent = '正在读取关联图片…'; $('#task-image-pagination').hidden = true;
    try {
      const query = new URLSearchParams({page, page_size:24, q:$('#task-image-search').value.trim(), state:$('#task-image-filter').value});
      const data = await request(`${url(project.id, task.id)}/images?${query}`);
      if (version !== imageEpoch) return;
      pages = data.pages; imageRecords = data.images; imageTotal = data.total;
      if (page > pages) return loadImages(pages);
      $('#task-image-state').textContent = data.total ? `共 ${data.total} 张匹配图片。点击图片进入标注工作台。` : task.image_count ? '没有匹配的图片，请调整搜索或状态筛选。' : '任务尚未关联图片。可先进入数据集导入图片，再返回同步新增图片。';
      for (const record of data.images) {
        const fragment = $('#task-image-template').content.cloneNode(true);
        const source = `/api/projects/${project.id}/datasets/${task.dataset_id}/images/${record.id}`;
        const image = fragment.querySelector('img'); image.src = `${source}/thumbnail`; image.alt = record.filename;
        const unavailable = fragment.querySelector('.task-image-unavailable');
        image.addEventListener('error', () => { image.hidden = true; unavailable.hidden = false; });
        fragment.querySelector('h4').textContent = record.filename;
        const badge = fragment.querySelector('.annotation-badge'); badge.textContent = stateNames[record.annotation_state]; badge.dataset.state = record.annotation_state;
        fragment.querySelector('p').textContent = `${record.width} × ${record.height}`;
        const button = fragment.querySelector('button'); button.setAttribute('aria-label', `标注任务图片：${record.filename}`);
        button.addEventListener('click', () => openRecord(project,task,record));
        $('#task-image-grid').append(fragment);
      }
      $('#task-image-pagination').hidden = !data.total;
      $('#task-image-page').textContent = `${page} / ${pages} 页`;
      $('#task-images-prev').disabled = page <= 1; $('#task-images-next').disabled = page >= pages;
      return data;
    } catch (error) { if (version === imageEpoch) $('#task-image-state').textContent = error.message; }
  }

  async function open(project, taskId = null) {
    const version = ++generation; imageEpoch += 1;
    if (lastParent !== project.id) { $('#task-search').value = ''; $('#task-sort').value = 'updated'; lastParent = project.id; }
    if (taskId && lastImageTask !== `${project.id}/${taskId}`) { $('#task-image-search').value=''; $('#task-image-filter').value=''; lastImageTask=`${project.id}/${taskId}`; }
    parent = { ...project }; current = null; ready = false; rows = [];
    $('#tasks-list-view').hidden = Boolean(taskId); $('#task-detail-view').hidden = !taskId;
    $('#task-detail-content').hidden = true; $('#task-detail-error').hidden = true;
    $('#task-detail-edit').disabled = true; $('#task-detail-delete').disabled = true; $('#create-task').disabled = true;
    $('#task-grid').replaceChildren(); $('#task-list-count').textContent = project.task_count;
    if (taskId) { $('#task-detail-name').textContent = '正在读取任务…'; $('#task-detail-description').textContent = ''; $('#task-back-link').href = `#/projects/${project.id}/tasks`; }
    else status('正在读取任务…', '');
    try {
      if (taskId) {
        const data = await request(url(project.id, taskId)); if (version !== generation) return;
        current = data.task;$('#task-detail-view .task-type-badge').textContent=current.task_type==='segment'?'实例分割':'目标检测';
        navigation.open(project.id,current.id,current.task_type);
        $('#task-source-details').open=Boolean(current.pending_image_count);
        $('#task-class-settings').open=!current.class_count;
        $('#task-class-summary').textContent=`${current.class_count} 个类别`;
        $('#task-detail-name').textContent = current.name; $('#task-detail-description').textContent = current.description || '暂无任务说明';
        $('#task-source-link').textContent = current.dataset_name; $('#task-source-link').href = `#/projects/${project.id}/datasets/${current.dataset_id}`;
        $('#task-bound-images').textContent = `${current.image_count} 张`;
        $('#task-source-images').textContent = `${current.source_image_count} 张`;
        $('#task-last-sync').textContent = date.format(new Date(current.last_synced_at));
        $('#task-pending-message').textContent = current.pending_image_count ? `数据集有 ${current.pending_image_count} 张新图片尚未关联到此任务。` : '任务图片清单与数据集当前原图一致。';
        $('#task-sync-images').disabled = !current.pending_image_count;
        $('#task-sync-feedback').textContent = '';
        $('#task-detail-edit').disabled = false; $('#task-detail-delete').disabled = false;
        $('#task-detail-content').hidden = false; onContext(current); await Promise.all([loadImages(1), annotations.open(project.id, current), preparation.open(project.id,current.id),versions.open(project.id,current.id),training.open(project.id,current.id)]);
      } else {
        const data = await request(url(project.id)); if (version !== generation) return;
        rows = data.tasks; ready = true; $('#create-task').disabled = false; render();
      }
    } catch (error) {
      if (version !== generation) return;
      if (taskId) { $('#task-detail-name').textContent = '无法打开任务'; $('#task-detail-error').textContent = error.message; $('#task-detail-error').hidden = false; }
      else status('暂时无法读取任务', error.message, () => open(project), '重新加载');
    }
  }

  function counters() {
    $('#task-name-counter').textContent = `${Array.from($('#task-name').value).length} / 60`;
    $('#task-description-counter').textContent = `${Array.from($('#task-description').value).length} / 1000`;
  }

  async function openEditor(record = null) {
    if (!parent || busy) return;
    editor = { project: { ...parent }, record: record ? { ...record } : null, loading: !record };
    form.reset(); $('#task-form-error').hidden = true;
    for (const field of ['name', 'description', 'dataset_id', 'task_type']) $(`#task-${field}-error`).textContent = '';
    $('#task-dialog-title').textContent = record ? '编辑任务' : '创建任务';
    $('#task-dialog-project').textContent = parent.name;
    $('#task-name').value = record?.name || ''; $('#task-description').value = record?.description || '';
    $('#task-type').value = record?.task_type || 'detect'; $('#task-type').disabled = Boolean(record);
    $('#task-source-select').replaceChildren();
    $('#task-source-select').disabled = true; $('#task-save').disabled = true;
    $('#task-save').textContent = record ? '保存修改' : '创建任务';
    $('#task-binding-note').textContent = record ? '任务类型和数据集已固定；如需更换，请创建新任务。' : '创建后类型与数据集固定；当前图片会关联到任务，后续新增图片可手动同步。';
    $('#task-no-dataset-link').href = `#/projects/${parent.id}`; $('#task-no-dataset-link').hidden = true;
    counters(); dialog.showModal(); $('#task-name').focus();
    if (record) {
      const option = document.createElement('option'); option.value = record.dataset_id; option.textContent = record.dataset_name; $('#task-source-select').append(option);
      $('#task-save').disabled = false; return;
    }
    const context = editor;
    try {
      const data = await request(`/api/projects/${context.project.id}/datasets`);
      if (editor !== context || !dialog.open) return;
      const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = data.datasets.length ? '请选择数据集' : '当前项目还没有数据集'; $('#task-source-select').append(placeholder);
      for (const dataset of data.datasets) { const option = document.createElement('option'); option.value = dataset.id; option.textContent = `${dataset.name} · ${dataset.image_count} 张图片`; $('#task-source-select').append(option); }
      context.loading = false;
      $('#task-source-select').disabled = !data.datasets.length; $('#task-save').disabled = !data.datasets.length;
      $('#task-no-dataset-link').hidden = Boolean(data.datasets.length);
    } catch (error) { if (editor === context && dialog.open) { $('#task-form-error').textContent = error.message; $('#task-form-error').hidden = false; } }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault(); if (busy || !editor || editor.loading) return;
    $('#task-form-error').hidden = true;
    for (const field of ['name', 'description', 'dataset_id', 'task_type']) $(`#task-${field}-error`).textContent = '';
    if (!$('#task-name').value.trim()) { $('#task-name-error').textContent = '请输入任务名称。'; $('#task-name').focus(); return; }
    if (!editor.record && !$('#task-source-select').value) { $('#task-dataset_id-error').textContent = '请选择关联数据集。'; return; }
    const context = editor;
    const values = { name: $('#task-name').value.trim(), description: $('#task-description').value.trim() };
    if (context.record) values.revision = context.record.revision;
    else { values.dataset_id = $('#task-source-select').value; values.task_type = $('#task-type').value; }
    busy = true; for (const field of form.querySelectorAll('button, input, textarea, select')) field.disabled = true; $('#task-save').textContent = '正在保存…';
    try {
      const data = await request(url(context.project.id, context.record?.id), { method: context.record ? 'PUT' : 'POST', body: values });
      dialog.close(); notify(context.record ? '任务修改已保存' : '任务已创建');
      if (!context.record) location.hash = `#/projects/${context.project.id}/tasks/${data.task.id}`;
      await onChanged();
    } catch (error) {
      $('#task-form-error').textContent = error.message; $('#task-form-error').hidden = false;
      for (const field of ['name', 'description', 'dataset_id', 'task_type']) $(`#task-${field}-error`).textContent = error.fields?.[field] || '';
    } finally {
      busy = false; for (const field of form.querySelectorAll('button, input, textarea, select')) field.disabled = false;
      $('#task-type').disabled = Boolean(context.record); $('#task-source-select').disabled = Boolean(context.record);
      $('#task-save').textContent = context.record ? '保存修改' : '创建任务';
    }
  });

  function openDeletion(record) {
    deletion = { project: { ...parent }, record: { ...record } }; $('#task-delete-form').reset();
    $('#task-delete-name').textContent = record.name;
    $('#task-delete-impact').textContent = `移除该任务及 ${record.image_count} 张图片的关联记录，数据集原图及其他任务会保留。同时移除本任务的类别、图片状态和工作区标注。如已有发布数据版本，删除将被阻止。`;
    $('#task-delete-error').hidden = true; $('#task-confirm-delete').disabled = true; deletionDialog.showModal(); $('#task-delete-confirmation').focus();
  }
  $('#task-delete-confirmation').addEventListener('input', () => { $('#task-confirm-delete').disabled = deleting || $('#task-delete-confirmation').value !== deletion?.record.name; });
  $('#task-delete-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (deleting || !deletion || $('#task-delete-confirmation').value !== deletion.record.name) return;
    const context = deletion; deleting = true; for (const field of $('#task-delete-form').querySelectorAll('button,input')) field.disabled = true;
    try {
      await request(url(context.project.id, context.record.id), { method: 'DELETE', body: { revision: context.record.revision, confirmation_name: $('#task-delete-confirmation').value } });
      deletionDialog.close(); notify('任务已删除，数据集原图已保留');
      if (location.hash.endsWith(`/tasks/${context.record.id}`)) location.hash = `#/projects/${context.project.id}/tasks`;
      await onChanged();
    } catch (error) { $('#task-delete-error').textContent = error.message; $('#task-delete-error').hidden = false; }
    finally { deleting = false; for (const field of $('#task-delete-form').querySelectorAll('button,input')) field.disabled = false; $('#task-confirm-delete').disabled = $('#task-delete-confirmation').value !== deletion?.record.name; }
  });
  $('#task-sync-images').addEventListener('click', async () => {
    if (syncing || !current) return;
    const project = { ...parent }, task = { ...current }; syncing = true; $('#task-sync-images').disabled = true;
    try { const result = await request(`${url(project.id, task.id)}/sync-images`, { method: 'POST', body: { revision: task.revision } }); await onChanged(); if (current?.id === task.id && parent?.id === project.id) $('#task-sync-feedback').textContent = `已同步 ${result.added_count} 张新增图片`; }
    catch (error) { if (current?.id === task.id && parent?.id === project.id) { $('#task-sync-feedback').textContent = error.message; $('#task-sync-images').disabled = false; } }
    finally { syncing = false; }
  });
  $('#create-task').addEventListener('click', () => openEditor());
  $('#task-detail-edit').addEventListener('click', () => { if (current) openEditor(current); });
  $('#task-detail-delete').addEventListener('click', () => { if (current) openDeletion(current); });
  $('#task-search').addEventListener('input', render); $('#task-sort').addEventListener('change', render);
  $('#tasks-refresh').addEventListener('click', onChanged);
  $('#task-name').addEventListener('input', counters); $('#task-description').addEventListener('input', counters);
  for (const id of ['task-close', 'task-cancel']) $(`#${id}`).addEventListener('click', () => { if (!busy) dialog.close(); });
  for (const id of ['task-delete-close', 'task-delete-cancel']) $(`#${id}`).addEventListener('click', () => { if (!deleting) deletionDialog.close(); });
  dialog.addEventListener('cancel', (event) => { if (busy) event.preventDefault(); }); deletionDialog.addEventListener('cancel', (event) => { if (deleting) event.preventDefault(); });
  $('#task-no-dataset-link').addEventListener('click', () => dialog.close());
  $('#task-images-prev').addEventListener('click', () => { if (page > 1) loadImages(page - 1); }); $('#task-images-next').addEventListener('click', () => { if (page < pages) loadImages(page + 1); });
  $('#task-image-refresh').addEventListener('click', () => loadImages(page));
  $('#task-image-filter-apply').addEventListener('click', () => loadImages(1));
  $('#task-image-filter').addEventListener('change', () => loadImages(1));
  $('#task-image-search').addEventListener('keydown', event => { if (event.key==='Enter') { event.preventDefault(); loadImages(1); } });
  return { open, canLeave: async () => !preparation.busy && !versions.busy && await annotations.canLeave(), hide() { versions.hide(); preparation.hide(); annotations.hide(); generation += 1; imageEpoch += 1; parent = null; current = null; ready = false; if ($('#task-image-preview').open) $('#task-image-preview').close(); } };
}
