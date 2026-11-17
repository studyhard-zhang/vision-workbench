import { createRoiEditor } from './roi-editor.js';
import { request } from './api.js';
import { createImageWorkspace } from './images.js';

const $ = (selector) => document.querySelector(selector);
const dateFormat = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', hour12: false,
});

export function createDatasetWorkspace({ notify, onChanged, onContext }) {
  const listView = $('#datasets-list-view');
  const itemView = $('#dataset-item-view');
  const grid = $('#dataset-grid');
  const state = $('#dataset-list-state');
  const dialog = $('#dataset-dialog');
  const form = $('#dataset-form');
  const name = $('#dataset-name');
  const description = $('#dataset-description');
  const deleteDialog = $('#dataset-delete-dialog');
  const deleteForm = $('#dataset-delete-form');
  let parent = null;
  let records = [];
  let current = null;
  const roiEditor=createRoiEditor({notify,onSaved:()=>open(parent,current.id)});
  let editor = null;
  let deletion = null;
  let busy = false;
  let deleting = false;
  let generation = 0;
  let lastProjectId = null;
  let listReady = false;
  const images = createImageWorkspace({ notify, onCountChange: (summary) => { if (current) current.image_count = summary.image_count; } });

  function stateMessage(title, message, action, label = '创建数据集') {
    state.hidden = false;
    state.replaceChildren();
    const symbol = document.createElement('div');
    symbol.className = 'dataset-state-icon';
    symbol.textContent = title.startsWith('正在') ? '…' : '▧';
    const heading = document.createElement('h3');
    heading.textContent = title;
    const text = document.createElement('p');
    text.textContent = message;
    state.append(symbol, heading, text);
    if (action) {
      const button = document.createElement('button');
      button.className = 'button primary';
      button.type = 'button';
      button.textContent = label;
      button.addEventListener('click', action);
      state.append(button);
    }
  }

  function renderList() {
    if (!listReady) return;
    const query = $('#dataset-search').value.trim().toLocaleLowerCase();
    const sort = $('#dataset-sort').value;
    const visible = records.filter((record) => `${record.name}\n${record.description}`.toLocaleLowerCase().includes(query));
    visible.sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name, 'zh-CN')
      : (sort === 'created' ? b.created_at.localeCompare(a.created_at) : b.updated_at.localeCompare(a.updated_at)));
    grid.replaceChildren();
    $('#dataset-list-count').textContent = records.length;
    $('#dataset-search-summary').hidden = !query;
    $('#dataset-search-summary').textContent = `找到 ${visible.length} 个数据集，共 ${records.length} 个`;
    if (!records.length) {
      stateMessage('创建第一个数据集', '按图片来源或采集批次整理数据。同一批原图可供不同任务复用。', () => openEditor());
      return;
    }
    if (!visible.length) {
      stateMessage('没有匹配的数据集', '试试其他名称或说明关键词。', () => {
        $('#dataset-search').value = '';
        renderList();
      }, '清除搜索');
      return;
    }
    state.hidden = true;
    for (const record of visible) {
      const fragment = $('#dataset-card-template').content.cloneNode(true);
      if(record.cover_image_id){
        const cover=document.createElement('img');cover.className='dataset-cover';cover.loading='lazy';cover.alt=`${record.name} 图片示例`;
        cover.src=`/api/projects/${parent.id}/datasets/${record.id}/images/${record.cover_image_id}/thumbnail`;
        cover.onerror=()=>cover.remove();fragment.querySelector('article').prepend(cover);
      }
      const link = fragment.querySelector('.dataset-name-link');
      link.textContent = record.name;
      link.href = `#/projects/${parent.id}/datasets/${record.id}`;
      const descriptionNode = fragment.querySelector('.dataset-card-description');
      fragment.querySelector('.dataset-image-count').textContent = `${record.image_count} 张图片 · ${record.task_count} 个任务`;
      descriptionNode.textContent = record.description || '暂无数据集说明';
      descriptionNode.classList.toggle('no-description', !record.description);
      const time = fragment.querySelector('time');
      time.dateTime = record.updated_at;
      time.textContent = dateFormat.format(new Date(record.updated_at));
      time.title = '最近更新（北京时间）';
      const editButton = fragment.querySelector('.edit-dataset');
      editButton.setAttribute('aria-label', `编辑数据集：${record.name}`);
      editButton.addEventListener('click', () => openEditor(record));
      const deleteButton = fragment.querySelector('.delete-dataset');
      deleteButton.setAttribute('aria-label', `删除数据集：${record.name}`);
      deleteButton.addEventListener('click', () => openDeletion(record));
      const openLink = fragment.querySelector('.open-dataset');
      openLink.href = `#/projects/${parent.id}/datasets/${record.id}`;
      openLink.setAttribute('aria-label', `查看数据集：${record.name}`);
      grid.append(fragment);
    }
  }

  async function open(project, datasetId = null) {
    const version = ++generation;
    if (lastProjectId !== project.id) {
      $('#dataset-search').value = '';
      $('#dataset-sort').value = 'updated';
      lastProjectId = project.id;
    }
    parent = { ...project };
    current = null;
    records = [];
    listReady = false;
    images.hide();
    listView.hidden = Boolean(datasetId);
    itemView.hidden = !datasetId;
    $('#dataset-load-error').hidden = true;
    $('#dataset-item-content').hidden = true;
    $('#dataset-edit-detail').disabled = true;
    $('#dataset-delete-detail').disabled = true;
    $('#create-dataset').disabled = true;
    grid.replaceChildren();
    $('#dataset-list-count').textContent = project.dataset_count;
    $('#dataset-search-summary').hidden = true;
    if (datasetId) {
      $('#dataset-item-name').textContent = '正在读取数据集…';
      $('#dataset-item-description').textContent = '';
      $('#dataset-back-link').href = `#/projects/${project.id}`;
    } else {
      stateMessage('正在读取数据集…', '');
    }
    try {
      if (datasetId) {
        const data = await request(`/api/projects/${project.id}/datasets/${datasetId}`);
        if (version !== generation) return;
        current = data.dataset;
        $('#dataset-item-name').textContent = current.name;
        $('#dataset-item-description').textContent = current.description || '暂无说明，可通过编辑数据集补充。';
        $('#dataset-parent-name').textContent = project.name;
        $('#dataset-item-created').textContent = dateFormat.format(new Date(current.created_at));
        $('#dataset-item-updated').textContent = dateFormat.format(new Date(current.updated_at));
        $('#dataset-item-content').hidden = false;
        $('#dataset-edit-detail').disabled = false;
        $('#dataset-delete-detail').disabled = false;
        onContext(current);
        await images.open(project, current);await roiEditor.open(project.id,current.id);
      } else {
        const data = await request(`/api/projects/${project.id}/datasets`);
        if (version !== generation) return;
        records = data.datasets;
        listReady = true;
        $('#create-dataset').disabled = false;
        renderList();
      }
    } catch (error) {
      if (version !== generation) return;
      if (datasetId) {
        $('#dataset-item-name').textContent = '无法打开数据集';
        $('#dataset-load-error').textContent = error.message;
        $('#dataset-load-error').hidden = false;
      } else {
        stateMessage('暂时无法读取数据集', error.message, () => open(project), '重新加载');
      }
    }
  }

  function updateCounters() {
    $('#dataset-name-counter').textContent = `${Array.from(name.value).length} / 60`;
    $('#dataset-description-counter').textContent = `${Array.from(description.value).length} / 1000`;
  }

  function openEditor(record = null) {
    if (!parent) return;
    editor = { parent: { ...parent }, record: record ? { ...record } : null };
    form.reset();
    $('#dataset-form-error').hidden = true;
    for (const field of ['name', 'description']) {
      $(`#dataset-${field}-error`).textContent = '';
      $(`#dataset-${field}`).removeAttribute('aria-invalid');
    }
    $('#dataset-dialog-title').textContent = record ? '编辑数据集' : '创建数据集';
    $('#dataset-save').textContent = record ? '保存修改' : '创建数据集';
    $('#dataset-dialog-parent').textContent = parent.name;
    name.value = record?.name || '';
    description.value = record?.description || '';
    updateCounters();
    dialog.showModal();
    name.focus();
  }

  function setBusy(value) {
    busy = value;
    form.setAttribute('aria-busy', String(value));
    for (const control of form.querySelectorAll('input, textarea, button')) control.disabled = value;
    $('#dataset-save').textContent = value ? '正在保存…' : editor?.record ? '保存修改' : '创建数据集';
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (busy || !editor) return;
    $('#dataset-form-error').hidden = true;
    for (const field of ['name', 'description']) {
      $(`#dataset-${field}-error`).textContent = '';
      $(`#dataset-${field}`).removeAttribute('aria-invalid');
    }
    if (!name.value.trim()) {
      $('#dataset-name-error').textContent = '请输入数据集名称。';
      name.setAttribute('aria-invalid', 'true');
      name.focus();
      return;
    }
    const { parent: project, record } = editor;
    setBusy(true);
    try {
      const values = { name: name.value.trim(), description: description.value.trim() };
      await request(`/api/projects/${project.id}/datasets${record ? `/${record.id}` : ''}`, {
        method: record ? 'PUT' : 'POST', body: record ? { ...values, revision: record.revision } : values,
      });
      dialog.close();
      notify(record ? '数据集修改已保存' : '数据集已创建');
      await onChanged();
    } catch (error) {
      $('#dataset-form-error').textContent = error.message;
      $('#dataset-form-error').hidden = false;
      for (const field of ['name', 'description']) {
        if (error.fields?.[field]) {
          $(`#dataset-${field}-error`).textContent = error.fields[field];
          $(`#dataset-${field}`).setAttribute('aria-invalid', 'true');
        }
      }
    } finally { setBusy(false); }
  });

  function openDeletion(record) {
    deletion = { parent: { ...parent }, record: { ...record } };
    deleteForm.reset();
    $('#dataset-delete-name').textContent = record.name;
    $('#dataset-delete-error').hidden = true;
    $('#dataset-confirm-delete').disabled = true;
    $('#dataset-delete-confirmation').disabled = record.image_count > 0 || record.task_count > 0;
    $('#dataset-delete-impact').textContent = record.task_count > 0 ? `数据集已关联 ${record.task_count} 个任务，请先处理这些任务。` : record.image_count > 0 ? `数据集包含 ${record.image_count} 张图片，不能直接删除，以免丢失原图和后续关联。` : '此空数据集尚无图片和任务，本次仅移除数据集及说明。';
    deleteDialog.showModal();
    $('#dataset-delete-confirmation').focus();
  }

  $('#dataset-delete-confirmation').addEventListener('input', () => {
    $('#dataset-confirm-delete').disabled = deleting || (deletion?.record.image_count || 0) > 0 || (deletion?.record.task_count || 0) > 0 || $('#dataset-delete-confirmation').value !== deletion?.record.name;
  });
  deleteForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (deleting || !deletion || deletion.record.image_count > 0 || deletion.record.task_count > 0 || $('#dataset-delete-confirmation').value !== deletion.record.name) return;
    const { parent: project, record } = deletion;
    deleting = true;
    for (const control of deleteForm.querySelectorAll('input, button')) control.disabled = true;
    $('#dataset-confirm-delete').textContent = '正在删除…';
    $('#dataset-delete-error').hidden = true;
    try {
      await request(`/api/projects/${project.id}/datasets/${record.id}`, { method: 'DELETE', body: {
        revision: record.revision, confirmation_name: $('#dataset-delete-confirmation').value,
      } });
      deleteDialog.close();
      if (location.hash.endsWith(`/datasets/${record.id}`)) location.hash = `#/projects/${project.id}`;
      notify('数据集已删除');
      await onChanged();
    } catch (error) {
      $('#dataset-delete-error').textContent = error.message;
      $('#dataset-delete-error').hidden = false;
    } finally {
      deleting = false;
      for (const control of deleteForm.querySelectorAll('input, button')) control.disabled = false;
      $('#dataset-confirm-delete').textContent = '确认删除';
      $('#dataset-confirm-delete').disabled = $('#dataset-delete-confirmation').value !== deletion?.record.name;
    }
  });

  $('#create-dataset').addEventListener('click', () => openEditor());
  $('#dataset-edit-detail').addEventListener('click', () => { if (current) openEditor(current); });
  $('#dataset-delete-detail').addEventListener('click', () => { if (current) openDeletion(current); });
  $('#dataset-search').addEventListener('input', renderList);
  $('#dataset-sort').addEventListener('change', renderList);
  $('#datasets-refresh').addEventListener('click', async () => {
    if (!parent) return;
    await onChanged();
  });
  name.addEventListener('input', updateCounters);
  description.addEventListener('input', updateCounters);
  for (const id of ['dataset-close', 'dataset-cancel']) $(`#${id}`).addEventListener('click', () => { if (!busy) dialog.close(); });
  for (const id of ['dataset-delete-close', 'dataset-delete-cancel']) $(`#${id}`).addEventListener('click', () => { if (!deleting) deleteDialog.close(); });
  dialog.addEventListener('cancel', (event) => { if (busy) event.preventDefault(); });
  deleteDialog.addEventListener('cancel', (event) => { if (deleting) event.preventDefault(); });
  return { open, hide() { generation += 1; parent = null; current = null; images.hide(); } };
}
