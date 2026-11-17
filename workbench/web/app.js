import { request } from './api.js';
import './maintenance.js';
import { createDatasetWorkspace } from './datasets.js';
import { createTaskWorkspace } from './tasks.js';

const $ = (selector) => document.querySelector(selector);
const listView = $('#list-view');
const detailView = $('#detail-view');
const grid = $('#project-grid');
const listState = $('#list-state');
const dialog = $('#project-dialog');
const deleteDialog = $('#delete-dialog');
const projectForm = $('#project-form');
const nameInput = $('#project-name');
const descriptionInput = $('#project-description');
const searchInput = $('#project-search');
const sortInput = $('#project-sort');
const saveButton = $('#save-project');
let projects = [];
let selectedProject = null;
let editingProject = null;
let deletingProject = null;
let listLoading = true;
let listError = null;
let formBusy = false;
let deleteBusy = false;
let listEpoch = 0;
let routeEpoch = 0;
let acceptedHash = location.hash;
let toastTimer;
const datasets = createDatasetWorkspace({
  notify: toast,
  onChanged: async () => { await refreshProjects(); await showRoute(); },
  onContext: (record) => {
    $('#detail-crumb').textContent = `/ ${selectedProject.name} / ${record.name}`;
    document.title = `${record.name} · ${selectedProject.name} · 视训工作台`;
  },
});
const tasks = createTaskWorkspace({
  notify: toast,
  onChanged: async () => { await refreshProjects(); await showRoute(); },
  onContext: (record) => {
    $('#detail-crumb').textContent = `/ ${selectedProject.name} / 任务 / ${record.name}`;
    document.title = `${record.name} · ${selectedProject.name} · 视训工作台`;
  },
});

const fullDate = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', hour12: false,
});
const shortDate = new Intl.DateTimeFormat('zh-CN', {
  timeZone: 'Asia/Shanghai', month: 'short', day: 'numeric',
});

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

function connection(online) {
  $('#connection-dot').className = `status-dot ${online ? 'online' : 'offline'}`;
  $('#connection-label').textContent = online ? '本地工作区已连接' : '工作区连接中断';
}

function toast(message) {
  clearTimeout(toastTimer);
  $('#toast-message').textContent = message;
  $('#toast').hidden = false;
  toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 4000);
}

function stateView({ title, description, symbol = 'folder', action, label }) {
  listState.hidden = false;
  listState.classList.remove('loading');
  listState.replaceChildren();
  const illustration = document.createElement('div');
  illustration.className = 'empty-illustration';
  illustration.append(icon(symbol));
  const heading = document.createElement('h2');
  heading.textContent = title;
  const text = document.createElement('p');
  text.textContent = description;
  listState.append(illustration, heading, text);
  if (action) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'button primary';
    button.append(icon(symbol === 'search' ? 'close' : symbol === 'refresh' ? 'refresh' : 'plus'), document.createTextNode(label));
    button.addEventListener('click', action);
    listState.append(button);
  }
}

function renderProjects() {
  grid.replaceChildren();
  $('#project-count').textContent = projects.length;
  $('#sidebar-count').textContent = projects.length;
  $('#search-summary').hidden = true;
  if (listLoading) {
    listState.hidden = false;
    listState.classList.add('loading');
    listState.replaceChildren();
    const ring = document.createElement('div');
    ring.className = 'loading-ring';
    listState.append(ring, document.createTextNode('正在读取项目…'));
    return;
  }
  if (listError) {
    stateView({ title: '暂时无法读取项目', description: listError.message, symbol: 'refresh', label: '重新连接', action: refreshProjects });
    return;
  }
  const query = searchInput.value.trim().toLocaleLowerCase();
  const visible = projects.filter((project) => `${project.name}\n${project.description}`.toLocaleLowerCase().includes(query));
  visible.sort((a, b) => sortInput.value === 'name'
    ? a.name.localeCompare(b.name, 'zh-CN')
    : sortInput.value === 'created'
      ? b.created_at.localeCompare(a.created_at)
      : b.updated_at.localeCompare(a.updated_at));
  if (!projects.length) {
    stateView({ title: '从第一个项目开始', description: '按业务场景组织项目，在项目内管理多个数据集，再为任务准备相应标注。', label: '创建第一个项目', action: () => openProjectDialog() });
    return;
  }
  if (query) {
    $('#search-summary').textContent = `找到 ${visible.length} 个项目，共 ${projects.length} 个`;
    $('#search-summary').hidden = false;
  }
  if (!visible.length) {
    stateView({ title: '没有找到匹配的项目', description: '试试其他关键词，或清除搜索后查看全部项目。', symbol: 'search', label: '清除搜索', action: () => { searchInput.value = ''; renderProjects(); searchInput.focus(); } });
    return;
  }
  listState.hidden = true;
  for (const project of visible) {
    const fragment = $('#project-card-template').content.cloneNode(true);
    const card = fragment.querySelector('article');
    card.querySelector('.project-dataset-count').textContent = `${project.dataset_count ?? 0} 个数据集 · ${project.task_count ?? 0} 个任务`;
    const link = card.querySelector('.project-name-link');
    link.textContent = project.name;
    link.href = `#/projects/${project.id}`;
    const description = card.querySelector('.card-description');
    description.textContent = project.description || '暂无项目说明';
    description.classList.toggle('no-description', !project.description);
    const time = card.querySelector('time');
    time.dateTime = project.updated_at;
    time.textContent = `${shortDate.format(new Date(project.updated_at))} 更新`;
    time.title = `${fullDate.format(new Date(project.updated_at))}（北京时间）`;
    card.querySelector('.edit-project').setAttribute('aria-label', `编辑项目：${project.name}`);
    card.querySelector('.edit-project').addEventListener('click', () => openProjectDialog(project));
    card.querySelector('.delete-project').setAttribute('aria-label', `删除项目：${project.name}`);
    card.querySelector('.delete-project').addEventListener('click', () => openDeleteDialog(project));
    const open = card.querySelector('.open-project');
    open.href = `#/projects/${project.id}`;
    open.setAttribute('aria-label', `进入项目：${project.name}`);
    grid.append(fragment);
  }
}

async function refreshProjects() {
  const epoch = ++listEpoch;
  listLoading = true;
  listError = null;
  $('#refresh-projects').disabled = true;
  renderProjects();
  try {
    const data = await request('/api/projects');
    if (epoch !== listEpoch) return;
    projects = data.projects;
    connection(true);
  } catch (error) {
    if (epoch !== listEpoch) return;
    listError = error;
    connection(false);
  } finally {
    if (epoch === listEpoch) {
      listLoading = false;
      $('#refresh-projects').disabled = false;
      renderProjects();
    }
  }
}

async function showRoute() {
  if (!await tasks.canLeave()) { history.replaceState(null, '', acceptedHash || '#/projects'); return; }
  acceptedHash = location.hash;
  const epoch = ++routeEpoch;
  const match = location.hash.match(/^#\/projects\/([0-9a-f]{32})(?:\/(datasets|tasks)(?:\/([0-9a-f]{32}))?)?$/);
  const isDetail = Boolean(match);
  detailView.classList.toggle('task-focused',Boolean(match && match[2]==='tasks' && match[3]));
  listView.hidden = isDetail;
  detailView.hidden = !isDetail;
  $('#detail-crumb').hidden = !isDetail;
  selectedProject = null;
  datasets.hide();
  tasks.hide();
  if (!isDetail) {
    document.title = '项目 · 视训工作台';
    renderProjects();
    return;
  }
  $('#detail-name').textContent = '正在读取项目…';
  $('#detail-description').textContent = '';
  $('#detail-crumb').textContent = '/ …';
  $('#detail-error').hidden = true;
  $('#detail-content').hidden = true;
  $('#edit-detail').disabled = true;
  $('#delete-detail').disabled = true;
  try {
    const { project } = await request(`/api/projects/${match[1]}`);
    if (epoch !== routeEpoch) return;
    selectedProject = project;
    $('#detail-name').textContent = project.name;
    $('#detail-description').textContent = project.description || '暂无项目说明，可通过编辑项目补充。';
    $('#detail-crumb').textContent = `/ ${project.name}`;
    $('#detail-created').textContent = fullDate.format(new Date(project.created_at));
    $('#detail-updated').textContent = fullDate.format(new Date(project.updated_at));
    $('#detail-created').title = '北京时间';
    $('#detail-updated').title = '北京时间';
    $('#detail-dataset-count').textContent = `${project.dataset_count} 个数据集`;
    $('#detail-task-count').textContent = `${project.task_count} 个任务`;
    $('#project-datasets-tab').href = `#/projects/${project.id}`;
    $('#project-tasks-tab').href = `#/projects/${project.id}/tasks`;
    $('#project-datasets-tab-count').textContent = project.dataset_count;
    $('#project-tasks-tab-count').textContent = project.task_count;
    const taskMode = match[2] === 'tasks';
    $('#project-datasets-tab').classList.toggle('selected', !taskMode);
    $('#project-tasks-tab').classList.toggle('selected', taskMode);
    $('#dataset-section').hidden = taskMode;
    $('#task-section').hidden = !taskMode;
    $('#detail-dataset-help').textContent = '各数据集保存原图，供后续任务关联使用。';
    $('#detail-content').hidden = false;
    $('#edit-detail').disabled = false;
    $('#delete-detail').disabled = false;
    document.title = `${project.name} · 视训工作台`;
    connection(true);
    if (taskMode) await tasks.open(project, match[3] || null);
    else await datasets.open(project, match[3] || null);
  } catch (error) {
    if (epoch !== routeEpoch) return;
    $('#detail-name').textContent = '无法打开项目';
    $('#detail-crumb').textContent = '/ 项目不可用';
    $('#detail-error').textContent = error.message;
    $('#detail-error').hidden = false;
    if (error.code === 'NETWORK_ERROR') connection(false);
  }
}

function updateCounters() {
  $('#name-counter').textContent = `${Array.from(nameInput.value).length} / 60`;
  $('#description-counter').textContent = `${Array.from(descriptionInput.value).length} / 1000`;
}

function resetFormErrors() {
  $('#form-error').hidden = true;
  for (const name of ['name', 'description']) {
    $(`#${name}-error`).textContent = '';
    $(`#project-${name}`).removeAttribute('aria-invalid');
  }
}

function openProjectDialog(project = null) {
  editingProject = project ? { ...project } : null;
  projectForm.reset();
  resetFormErrors();
  nameInput.value = project?.name || '';
  descriptionInput.value = project?.description || '';
  $('#project-dialog-title').textContent = project ? '编辑项目' : '创建项目';
  $('#project-dialog-description').textContent = project ? '更新项目名称和说明，已保存的数据不会因重命名而改变。' : '按业务场景创建项目，在项目下管理多个数据集与任务。';
  saveButton.textContent = project ? '保存修改' : '创建项目';
  updateCounters();
  dialog.showModal();
  nameInput.focus();
}

function setFormBusy(busy) {
  formBusy = busy;
  projectForm.setAttribute('aria-busy', String(busy));
  for (const control of projectForm.querySelectorAll('button, input, textarea')) control.disabled = busy;
  saveButton.textContent = busy ? '正在保存…' : editingProject ? '保存修改' : '创建项目';
}

projectForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (formBusy) return;
  resetFormErrors();
  if (!nameInput.value.trim()) {
    $('#name-error').textContent = '请输入项目名称。';
    nameInput.setAttribute('aria-invalid', 'true');
    nameInput.focus();
    return;
  }
  setFormBusy(true);
  const editing = editingProject;
  try {
    const values = { name: nameInput.value.trim(), description: descriptionInput.value.trim() };
    const data = await request(editing ? `/api/projects/${editing.id}` : '/api/projects', {
      method: editing ? 'PUT' : 'POST', body: editing ? { ...values, revision: editing.revision } : values,
    });
    dialog.close();
    toast(editing ? '项目修改已保存' : '项目已创建');
    if (!editing) location.hash = `#/projects/${data.project.id}`;
    await refreshProjects();
    await showRoute();
  } catch (error) {
    $('#form-error').textContent = error.message;
    $('#form-error').hidden = false;
    for (const name of ['name', 'description']) {
      if (error.fields?.[name]) {
        $(`#${name}-error`).textContent = error.fields[name];
        $(`#project-${name}`).setAttribute('aria-invalid', 'true');
      }
    }
    if (error.code === 'NETWORK_ERROR') connection(false);
  } finally { setFormBusy(false); }
});

function openDeleteDialog(project) {
  deletingProject = { ...project };
  $('#delete-form').reset();
  $('#delete-project-name').textContent = project.name;
  $('#confirmation-error').textContent = '';
  $('#delete-error').hidden = true;
  const hasDatasets = (project.dataset_count || 0) > 0;
  $('#delete-confirmation').disabled = hasDatasets;
  $('#delete-impact-message').textContent = hasDatasets
    ? `项目包含 ${project.dataset_count} 个数据集，请先处理这些数据集后再删除项目。`
    : '本次将移除该空项目及其说明。';
  $('#confirm-delete').disabled = true;
  deleteDialog.showModal();
  request(`/api/projects/${project.id}/delete-impact`).then(impact=>{
    if(deletingProject?.id!==project.id||!deleteDialog.open)return;
    const c=impact.counts;
    $('#delete-impact-message').textContent=`影响范围：${c.datasets}个数据集、${c.images}张原图、${c.vision_tasks}个任务、${c.task_data_versions}个冻结版本、${c.training_jobs}条训练、${c.model_versions}个模型。${impact.notice}`;
  }).catch(()=>{});
  $('#delete-confirmation').focus();
}

$('#delete-confirmation').addEventListener('input', () => {
  $('#confirm-delete').disabled = deleteBusy || (deletingProject?.dataset_count || 0) > 0 || $('#delete-confirmation').value !== deletingProject?.name;
  $('#confirmation-error').textContent = '';
});

$('#delete-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (deleteBusy || !deletingProject || (deletingProject.dataset_count || 0) > 0) return;
  if ($('#delete-confirmation').value !== deletingProject.name) return;
  deleteBusy = true;
  for (const control of $('#delete-form').querySelectorAll('button, input')) control.disabled = true;
  $('#confirm-delete').textContent = '正在删除…';
  $('#delete-error').hidden = true;
  try {
    await request(`/api/projects/${deletingProject.id}`, { method: 'DELETE', body: {
      revision: deletingProject.revision, confirmation_name: $('#delete-confirmation').value,
    } });
    deleteDialog.close();
    if (location.hash.includes(deletingProject.id)) location.hash = '#/projects';
    toast('项目已删除');
    await refreshProjects();
    await showRoute();
  } catch (error) {
    $('#delete-error').textContent = error.message;
    $('#delete-error').hidden = false;
    $('#confirmation-error').textContent = error.fields?.confirmation_name || '';
    if (error.code === 'NETWORK_ERROR') connection(false);
  } finally {
    deleteBusy = false;
    for (const control of $('#delete-form').querySelectorAll('button, input')) control.disabled = false;
    $('#confirm-delete').disabled = $('#delete-confirmation').value !== deletingProject?.name;
    $('#confirm-delete').textContent = '确认删除';
  }
});

$('#create-project').addEventListener('click', () => openProjectDialog());
$('#edit-detail').addEventListener('click', () => { if (selectedProject) openProjectDialog(selectedProject); });
$('#delete-detail').addEventListener('click', () => { if (selectedProject) openDeleteDialog(selectedProject); });
for (const id of ['close-project-dialog', 'cancel-project']) $( `#${id}`).addEventListener('click', () => { if (!formBusy) dialog.close(); });
for (const id of ['close-delete-dialog', 'cancel-delete']) $( `#${id}`).addEventListener('click', () => { if (!deleteBusy) deleteDialog.close(); });
dialog.addEventListener('cancel', (event) => { if (formBusy) event.preventDefault(); });
deleteDialog.addEventListener('cancel', (event) => { if (deleteBusy) event.preventDefault(); });
nameInput.addEventListener('input', updateCounters);
descriptionInput.addEventListener('input', updateCounters);
searchInput.addEventListener('input', renderProjects);
sortInput.addEventListener('change', renderProjects);
$('#refresh-projects').addEventListener('click', async () => { await refreshProjects(); await showRoute(); });
window.addEventListener('hashchange', showRoute);
if (!location.hash) history.replaceState(null, '', '#/projects');
await refreshProjects();
await showRoute();
