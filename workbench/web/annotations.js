import { request } from './api.js';
import { createBoxEditor } from './box-editor.js';
const $ = (selector) => document.querySelector(selector);
export const stateNames = { unannotated: '未标注', annotated: '已标注', no_target: '确认无目标', excluded: '排除' };

export function createAnnotationWorkspace({ notify, onChanged }) {
  let context = null, epoch = 0, classes = [], editing = null, working = false;
  let classInitial = '';
  const boxEditor = createBoxEditor({ beforeLeave: canLeave, confirmClear: message => confirmDiscard(message, '清除当前标注', '清除标注'), notify });
  const classDialog = $('#class-dialog'), form = $('#class-form'), previewDialog = $('#task-image-preview');
  const base = (ctx) => `/api/projects/${ctx.projectId}/tasks/${ctx.taskId}`;
  const classDirty = () => classDialog.open && !editing?.deleting && classInitial !== $('#class-name').value;
  const stateDirty = () => previewDialog.open && boxEditor.dirty;
  const dirty = () => classDirty() || stateDirty();

  let discardResolve = null, discardPromise = null;
  function confirmDiscard(message = '有尚未保存的修改。放弃后将恢复到上次保存的内容。', title = '确认放弃修改', label = '放弃修改') {
    if (discardPromise) return discardPromise;
    $('#discard-title').textContent = title; $('#discard-confirm').textContent = label;
    $('#discard-message').textContent = message;
    $('#discard-dialog').showModal();
    discardPromise = new Promise((resolve) => { discardResolve = resolve; });
    return discardPromise;
  }
  function resolveDiscard(accepted) { $('#discard-dialog').close(); discardResolve?.(accepted); discardResolve = null; discardPromise = null; }
  $('#discard-stay').onclick = () => resolveDiscard(false);
  $('#discard-confirm').onclick = () => resolveDiscard(true);
  $('#discard-dialog').addEventListener('cancel', (event) => { event.preventDefault(); resolveDiscard(false); });
  async function canLeave() {
    if (working || boxEditor.busy) return false;
    return !dirty() || await confirmDiscard();
  }
  async function closeClass() { if (await canLeave()) classDialog.close(); }
  async function closePreview() { if (await canLeave()) { const reload=boxEditor.changed; previewDialog.close(); if (reload) await onChanged(); } }

  async function open(projectId, task) {
    const version = ++epoch;
    context = { projectId, taskId: task.id, taskType:task.task_type };
    $('#class-list').replaceChildren(); $('#class-status').textContent = '正在读取任务类别…';
    $('#class-add').disabled = true;
    const unannotated = task.image_count - task.annotated_count - task.no_target_count - task.excluded_count;
    $('#annotation-summary').textContent = `未标注 ${unannotated} · 已标注 ${task.annotated_count} · 确认无目标 ${task.no_target_count} · 排除 ${task.excluded_count}`;
    try {
      const data = await request(`${base(context)}/classes`);
      if (version !== epoch) return;
      classes = data.classes;
      $('#class-count').textContent = `${classes.length} 个类别`;
      $('#class-status').textContent = classes.length ? '' : '还没有类别。请先添加本任务需要识别的对象，例如“划痕”“凹陷”。';
      $('#class-add').disabled = false;
      for (const record of classes) {
        const item = document.createElement('div'); item.className = 'class-item';
        const swatch = document.createElementNS('http://www.w3.org/2000/svg','svg'); swatch.classList.add('class-swatch'); swatch.setAttribute('viewBox','0 0 18 18'); swatch.setAttribute('aria-hidden','true');
        const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');rect.setAttribute('width','18');rect.setAttribute('height','18');rect.setAttribute('rx','3');rect.setAttribute('fill',record.color);swatch.append(rect);
        const name = document.createElement('strong'); name.textContent = record.name;
        const count = document.createElement('span'); count.textContent = `${record.box_count} 个${context.taskType==='segment'?'多边形':'框'}`;
        const edit = document.createElement('button'); edit.className = 'button secondary'; edit.textContent = '编辑'; edit.setAttribute('aria-label', `编辑类别：${record.name}`); edit.onclick = () => openClass(record);
        const remove = document.createElement('button'); remove.className = 'button secondary'; remove.textContent = '删除'; remove.setAttribute('aria-label', `删除类别：${record.name}`); remove.onclick = () => openClass(record, true);
        item.append(swatch,name,count,edit,remove); $('#class-list').append(item);
      }
    } catch (error) { if (version === epoch) $('#class-status').textContent = error.message; }
  }

  function openClass(record = null, deleting = false) {
    if (!context || working) return;
    editing = { ...context, record, deleting };
    form.reset(); $('#class-error').hidden = true;
    $('#class-title').textContent = deleting ? '删除类别' : record ? '编辑类别' : '添加类别';
    $('#class-name').value = record?.name || '';
    $('#class-fields').hidden = deleting; $('#class-delete-fields').hidden = !deleting;
    $('#class-delete-description').textContent = deleting ? `请输入“${record.name}”以确认删除。已有标注引用的类别不能删除。` : '';
    $('#class-submit').textContent = deleting ? '确认删除' : '保存类别';
    $('#class-submit').className = `button ${deleting ? 'danger' : 'primary'}`;
    $('#class-submit').disabled = deleting;
    classInitial = $('#class-name').value;
    classDialog.showModal(); $(deleting ? '#class-confirmation' : '#class-name').focus();
  }
  $('#class-add').onclick = () => openClass();
  $('#class-refresh').onclick = onChanged;
  $('#class-confirmation').oninput = () => { $('#class-submit').disabled = working || $('#class-confirmation').value !== editing?.record?.name; };
  $('#class-close').onclick = closeClass; $('#class-cancel').onclick = closeClass;
  classDialog.addEventListener('cancel', (event) => { event.preventDefault(); closeClass(); });
  form.addEventListener('submit', async (event) => {
    event.preventDefault(); if (working || !editing) return;
    const ctx = editing, record = ctx.record;
    const body = ctx.deleting ? { revision: record.revision, confirmation_name: $('#class-confirmation').value } : { name: $('#class-name').value, ...(record ? { revision: record.revision } : {}) };
    working = true; $('#class-error').hidden = true;
    for (const node of form.querySelectorAll('input,button')) node.disabled = true;
    try {
      await request(`${base(ctx)}/classes${record ? `/${record.id}` : ''}`, { method: ctx.deleting ? 'DELETE' : record ? 'PUT' : 'POST', body });
      classDialog.close(); working = false; notify(ctx.deleting ? '类别已删除' : '类别已保存'); await onChanged();
    } catch (error) { $('#class-error').textContent = error.message; $('#class-error').hidden = false; }
    finally { working = false; for (const node of form.querySelectorAll('input,button')) node.disabled = false; $('#class-submit').disabled = ctx.deleting && $('#class-confirmation').value !== record.name; }
  });

  function openImage(projectId, task, record, navigation) {
    $('#task-preview-title').textContent=record.filename;
    $('#task-preview-info').textContent=task.name;
    previewDialog.showModal();
    boxEditor.open({ projectId, task, record, navigation });
  }
  $('#task-preview-close').onclick = closePreview;
  previewDialog.addEventListener('cancel', (event) => { event.preventDefault(); closePreview(); });
  previewDialog.addEventListener('close', () => { boxEditor.hide(); });
  window.addEventListener('beforeunload', (event) => { if (dirty() || working || boxEditor.busy) { event.preventDefault(); event.returnValue = ''; } });
  return { open, openImage, canLeave, hide() { epoch++; context = null; boxEditor.hide(); classDialog.close(); previewDialog.close(); } };
}
