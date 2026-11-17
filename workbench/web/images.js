import { request, ApiError } from './api.js';
const $ = (selector) => document.querySelector(selector);
export const fileSize = (bytes) => bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

export function createImageWorkspace({ notify, onCountChange }) {
  let context = null, records = [], page = 1, pages = 1, epoch = 0, searchTimer;
  let uploadContext = null, entries = [], uploading = false, stopRequested = false;
  let viewerContext = null, viewerList = [], viewerIndex = 0, viewerEpoch = 0, groupBusy = false;
  const importDialog = $('#image-import-dialog');
  const viewer = $('#image-viewer');
  const base = (scope) => `/api/projects/${scope.projectId}/datasets/${scope.datasetId}/images`;
  const same = (a, b) => a && b && a.projectId === b.projectId && a.datasetId === b.datasetId;

  function message(title, detail, retry = false) {
    const state = $('#images-state');
    state.hidden = false;
    state.replaceChildren();
    const heading = document.createElement('h3'); heading.textContent = title;
    const text = document.createElement('p'); text.textContent = detail;
    state.append(heading, text);
    if (retry) {
      const button = document.createElement('button'); button.className = 'button secondary'; button.textContent = '重新加载';
      button.addEventListener('click', () => load(page)); state.append(button);
    }
  }

  async function load(nextPage = 1) {
    if (!context) return;
    const scope = { ...context }, version = ++epoch;
    page = nextPage;
    $('#image-grid').replaceChildren();
    $('#image-pagination').hidden = true;
    message('正在读取图片…', '');
    try {
      const query = new URLSearchParams({ page: String(page), page_size: '48', q: $('#image-search').value.trim(), sort: $('#image-sort').value });
      const data = await request(`${base(scope)}?${query}`);
      if (version !== epoch || !same(scope, context)) return;
      records = data.images; pages = data.pages;
      if (page > pages) return load(pages);
      $('#image-total').textContent = data.summary.image_count;
      $('#image-total-bytes').textContent = data.summary.original_bytes ? fileSize(data.summary.original_bytes) : '0 KB';
      $('#image-group-count').textContent = data.summary.group_count;
      $('#image-results-label').textContent = data.total === data.summary.image_count ? `共 ${data.total} 张` : `匹配 ${data.total} / ${data.summary.image_count} 张`;
      onCountChange(data.summary);
      if (!records.length) {
        message(data.summary.image_count ? '没有匹配的图片' : '导入第一批图片', data.summary.image_count ? '尝试其他文件名关键词。' : '选择 JPG、JPEG 或 PNG 图片，按来源整理你的原图数据。');
        return;
      }
      $('#images-state').hidden = true;
      records.forEach((record, index) => {
        const fragment = $('#image-card-template').content.cloneNode(true);
        const button = fragment.querySelector('.image-open');
        button.setAttribute('aria-label', `查看图片：${record.filename}`);
        const image = fragment.querySelector('img');
        image.src = `${base(scope)}/${record.id}/thumbnail`;
        image.alt = record.filename;
        image.addEventListener('error', () => { image.hidden = true; button.querySelector('.thumbnail-error').hidden = false; });
        button.addEventListener('click', () => openViewer(index));
        fragment.querySelector('.image-filename').textContent = record.filename;
        fragment.querySelector('.image-filename').title = record.filename;
        fragment.querySelector('.image-dimensions').textContent = `${record.width} × ${record.height} · ${record.format}`;
        fragment.querySelector('.image-group').hidden = true;
        $('#image-grid').append(fragment);
      });
      $('#image-pagination').hidden = false;
      $('#image-page-label').textContent = `${page} / ${pages} 页`;
      $('#images-prev').disabled = page <= 1;
      $('#images-next').disabled = page >= pages;
    } catch (error) {
      if (version === epoch && same(scope, context)) message('暂时无法读取图片', error.message, true);
    }
  }

  function renderEntries() {
    const counts = { imported: 0, duplicate: 0, failed: 0, pending: 0, processing: 0 };
    const names = { imported: '成功', duplicate: '重复', failed: '失败', pending: '待导入', processing: '处理中' };
    $('#import-file-list').replaceChildren();
    for (const entry of entries) {
      counts[entry.status] += 1;
      const row = document.createElement('li'); row.className = `import-file-row ${entry.status}`;
      const info = document.createElement('div');
      const name = document.createElement('strong'); name.textContent = entry.file.name;
      const detail = document.createElement('span'); detail.textContent = entry.message || fileSize(entry.file.size);
      info.append(name, detail);
      const status = document.createElement('span'); status.className = 'import-status'; status.textContent = names[entry.status];
      row.append(info, status); $('#import-file-list').append(row);
    }
    $('#import-summary').textContent = entries.length ? `共 ${entries.length} 张 · 成功 ${counts.imported} · 重复 ${counts.duplicate} · 失败 ${counts.failed} · 待处理 ${counts.pending + counts.processing}` : '还未选择图片';
    $('#import-start').disabled = uploading || counts.pending === 0;
    $('#import-start').textContent = uploading ? '正在导入…' : counts.imported || counts.failed || counts.duplicate ? '继续导入未处理项' : '开始导入';
    $('#import-retry').hidden = !counts.failed || uploading;
    $('#import-stop').hidden = !uploading;
    $('#import-stop').disabled = stopRequested;
    $('#import-stop').textContent = stopRequested ? '当前图片处理完成后停止' : '停止后续导入';
    for (const id of ['import-choose', 'import-group', 'import-close', 'import-done']) $(`#${id}`).disabled = uploading;
  }

  function selectFiles(files) {
    if (uploading) return;
    const list = [...files];
    $('#import-error').hidden = true;
    if (list.length > 200) {
      $('#import-error').textContent = '每批最多 200 张，请减少选择后重试。'; $('#import-error').hidden = false; return;
    }
    entries = list.map((file) => {
      const invalid = !/\.(jpe?g|png)$/i.test(file.name) ? '仅支持 JPG/JPEG、PNG。' : !file.size || file.size > 20 * 1048576 ? '图片不能为空，单张最多 20MB。' : '';
      return { file, status: invalid ? 'failed' : 'pending', message: invalid };
    });
    renderEntries();
  }

  function openImport() {
    if (!context || uploading) return;
    uploadContext = { ...context }; entries = []; stopRequested = false;
    $('#import-dataset-name').textContent = context.datasetName;
    $('#image-file-input').value = '';
    $('#import-group').value = '';
    $('#import-error').hidden = true;
    renderEntries(); importDialog.showModal();
  }

  async function uploadBatch() {
    if (uploading || !uploadContext) return;
    const group = $('#import-group').value.trim();
    if (Array.from(group).length > 80 || /[\x00-\x1f]/.test(group)) {
      $('#import-error').textContent = '采集组最多 80 个字符，不能包含控制字符。'; $('#import-error').hidden = false; return;
    }
    const scope = { ...uploadContext };
    uploading = true; stopRequested = false; $('#import-error').hidden = true; renderEntries();
    try {
      for (const entry of entries) {
        if (stopRequested) break;
        if (entry.status !== 'pending') continue;
        entry.status = 'processing'; entry.message = '正在传输与校验…'; renderEntries();
        try {
          const response = await fetch(base(scope), { method: 'POST', headers: {
            'Content-Type': entry.file.type || 'application/octet-stream',
            'X-File-Name': encodeURIComponent(entry.file.name), 'X-Group-Id': encodeURIComponent(group),
          }, body: entry.file });
          let data;
          try { data = await response.json(); } catch { throw new ApiError('响应无法识别，请刷新检查后重试。', 'INVALID_RESPONSE'); }
          if (!response.ok) throw new ApiError(data.error?.message || '导入失败，请重试。', data.error?.code);
          entry.status = data.status; entry.message = data.message;
        } catch (error) {
          entry.status = 'failed';
          entry.message = error.message || '连接中断，请刷新检查或重试。相同内容会自动跳过。';
        }
        renderEntries();
      }
    } finally {
      uploading = false; renderEntries();
      if (same(scope, context)) await load(1);
    }
  }

  async function openViewer(index, list = records) {
    viewerList = [...list]; viewerIndex = index;
    viewerContext = { ...context };
    if (!viewer.open) viewer.showModal();
    await showViewer();
  }

  async function showViewer() {
    const version = ++viewerEpoch, scope = { ...viewerContext }, current = viewerList[viewerIndex];
    $('#image-view-title').textContent = current.filename;
    $('#image-view-error').hidden = true;
    $('#image-group-feedback').textContent = '';
    $('#image-large').hidden = true;
    $('#image-view-status').textContent = '正在读取图片…';
    $('#image-view-info').textContent = `${current.width} × ${current.height} · ${current.format} · ${fileSize(current.byte_size)}`;
    $('#image-view-position').textContent = `当前页 ${viewerIndex + 1} / ${viewerList.length}`;
    $('#image-view-prev').disabled = viewerIndex === 0;
    $('#image-view-next').disabled = viewerIndex === viewerList.length - 1;
    viewer.classList.remove('actual-size'); $('#image-zoom').textContent = '查看 1:1';
    $('#image-group-save').disabled = true;
    $('#image-view-group').disabled = true;
    $('#image-original-link').removeAttribute('href');
    try {
      const data = await request(`${base(scope)}/${current.id}`);
      if (version !== viewerEpoch) return;
      viewerList[viewerIndex] = data.image;
      $('#image-view-group').value = data.image.group_id;
      $('#image-original-link').href = `${base(scope)}/${current.id}/original`;
      const image = $('#image-large');
      image.onload = () => { if (version === viewerEpoch) { image.hidden = false; $('#image-view-status').textContent = ''; } };
      image.onerror = () => { if (version === viewerEpoch) $('#image-view-status').textContent = '预览加载失败，可关闭后重试或下载原图检查。'; };
      image.src = `${base(scope)}/${current.id}/view`;
      image.alt = current.filename;
      $('#image-group-save').disabled = false;
      $('#image-view-group').disabled = false;
    } catch (error) {
      if (version === viewerEpoch) $('#image-view-status').textContent = error.message;
    }
  }

  $('#image-group-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (groupBusy || !viewerContext) return;
    const current = viewerList[viewerIndex], scope = { ...viewerContext };
    groupBusy = true; $('#image-view-error').hidden = true;
    $('#image-group-feedback').textContent = '';
    $('#image-group-save').textContent = '正在保存…';
    for (const id of ['image-group-save', 'image-view-group', 'image-view-prev', 'image-view-next', 'image-view-close']) $(`#${id}`).disabled = true;
    try {
      const data = await request(`${base(scope)}/${current.id}`, { method: 'PUT', body: { group_id: $('#image-view-group').value.trim(), revision: current.revision } });
      viewerList[viewerIndex] = data.image;
      $('#image-view-group').value = data.image.group_id;
      $('#image-group-feedback').textContent = '采集组已保存';
      if (same(scope, context)) await load(page);
    } catch (error) { $('#image-view-error').textContent = error.message; $('#image-view-error').hidden = false; }
    finally {
      groupBusy = false;
      $('#image-group-save').textContent = '保存采集组';
      for (const id of ['image-group-save', 'image-view-group', 'image-view-close']) $(`#${id}`).disabled = false;
      $('#image-view-prev').disabled = viewerIndex === 0;
      $('#image-view-next').disabled = viewerIndex === viewerList.length - 1;
    }
  });

  $('#open-image-import').addEventListener('click', openImport);
  $('#import-choose').addEventListener('click', () => $('#image-file-input').click());
  $('#image-file-input').addEventListener('change', (event) => selectFiles(event.target.files));
  $('#image-dropzone').addEventListener('dragover', (event) => { event.preventDefault(); if (!uploading) $('#image-dropzone').classList.add('dragover'); });
  $('#image-dropzone').addEventListener('dragleave', () => $('#image-dropzone').classList.remove('dragover'));
  $('#image-dropzone').addEventListener('drop', (event) => { event.preventDefault(); $('#image-dropzone').classList.remove('dragover'); selectFiles(event.dataTransfer.files); });
  $('#import-start').addEventListener('click', uploadBatch);
  $('#import-retry').addEventListener('click', () => { for (const entry of entries) if (entry.status === 'failed') { entry.status = 'pending'; entry.message = ''; } uploadBatch(); });
  $('#import-stop').addEventListener('click', () => { stopRequested = true; renderEntries(); });
  for (const id of ['import-close', 'import-done']) $(`#${id}`).addEventListener('click', () => { if (!uploading) importDialog.close(); });
  importDialog.addEventListener('cancel', (event) => { if (uploading) event.preventDefault(); });
  window.addEventListener('beforeunload', (event) => { if (uploading) { event.preventDefault(); event.returnValue = ''; } });
  $('#image-search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => load(1), 250); });
  $('#image-sort').addEventListener('change', () => load(1));
  $('#images-refresh').addEventListener('click', () => load(page));
  $('#images-prev').addEventListener('click', () => { if (page > 1) load(page - 1); });
  $('#images-next').addEventListener('click', () => { if (page < pages) load(page + 1); });
  $('#image-view-prev').addEventListener('click', () => { if (!groupBusy && viewerIndex > 0) { viewerIndex -= 1; showViewer(); } });
  $('#image-view-next').addEventListener('click', () => { if (!groupBusy && viewerIndex < viewerList.length - 1) { viewerIndex += 1; showViewer(); } });
  $('#image-view-close').addEventListener('click', () => { if (!groupBusy) { viewerEpoch += 1; viewer.close(); } });
  viewer.addEventListener('cancel', (event) => { if (groupBusy) event.preventDefault(); else viewerEpoch += 1; });
  $('#image-zoom').addEventListener('click', () => { const actual = viewer.classList.toggle('actual-size'); $('#image-zoom').textContent = actual ? '适应窗口' : '查看 1:1'; });
  return {
    async open(project, dataset) {
      const changed = !same(context, { projectId: project.id, datasetId: dataset.id });
      context = { projectId: project.id, datasetId: dataset.id, datasetName: dataset.name };
      $('#image-total').textContent = dataset.image_count;
      $('#image-total-bytes').textContent = '—';
      $('#image-group-count').textContent = '—';
      $('#image-results-label').textContent = '正在读取';
      if (changed) { $('#image-search').value = ''; $('#image-sort').value = 'newest'; }
      await load(1);
    },
    hide() {
      epoch += 1; viewerEpoch += 1; context = null; clearTimeout(searchTimer);
      if (viewer.open && !groupBusy) viewer.close();
      if (uploading) stopRequested = true; else if (importDialog.open) importDialog.close();
    },
  };
}
