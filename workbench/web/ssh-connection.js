import {request} from './api.js';

const $ = selector => document.querySelector(selector);
const endpoint = '/api/remote/instances';

export function createSSHConnection({onSaved}) {
  const dialog = $('#ssh-dialog');
  const form = $('#ssh-form');
  let busy = false;
  let key = null;
  let candidate = null;
  let activeId = null;
  let timer;

  function issue(message) {
    $('#ssh-error').textContent = message;
    $('#ssh-error').hidden = !message;
  }

  function sync() {
    const preparing = ['preparing','validating'].includes(candidate?.state);
    for(const id of ['ssh-kind','ssh-python','ssh-workspace','ssh-check-local'])$('#'+id).disabled=busy||preparing;
    for (const field of form.querySelectorAll('input, button,select')) field.disabled = busy || preparing;
    $('#ssh-save').disabled = busy || preparing || !key || !$('#ssh-confirm').checked;
    $('#ssh-prepare').disabled = busy || preparing || Boolean(candidate?.plan.blockers.length || candidate?.plan.missing.length || candidate?.plan.missing_weights.length);
    $('#ssh-enable-candidate').disabled = busy;
    const selected = $('#ssh-instance-select').value;
    $('#ssh-activate').disabled = busy || preparing || !selected || selected === activeId;
  }

  function clearKey() {
    key = null;
    $('#ssh-fingerprint').textContent = '';
    $('#ssh-confirm').checked = false;
    $('#ssh-confirm-label').hidden = true;
    sync();
  }

  function render(state) {
    candidate = state.candidate;
    activeId = state.active;
    const select = $('#ssh-instance-select');
    const previous = select.value;
    select.replaceChildren(...state.available.map(item => {
      const label = item.id === state.active ? `${item.name}（当前）` : item.name;
      return new Option(label, item.id);
    }));
    const selected = [...select.options].find(item => item.value === previous);
    select.value = selected?.value ?? state.active;
    $('#ssh-instances').hidden = !state.available.length;
    $('#ssh-environment').hidden = !candidate;
    if (candidate) {
      const {plan, state: phase} = candidate;
      const free = Number.isFinite(plan.free_bytes) ? ` · 可用 ${(plan.free_bytes/1024**3).toFixed(1)} GiB` : '';
      $('#ssh-environment-summary').textContent = `${candidate.endpoint} · ${plan.gpu?.name ?? 'GPU 待核对'}${plan.python ? ' · Python '+plan.python : ''}${free}`;
      const rows = [...plan.blockers];
      if(plan.executable)rows.unshift('当前校验Python：'+plan.executable);
      if (phase === 'ready') {
        rows.push('已有环境及已准备的模型通过CUDA、NMS、加载和前向检查。训练时仍会核对资源和参数。');
      } else {
        for (const item of plan.missing) rows.push(`${item.name} · ${item.installed ? '当前 '+item.installed : '未安装'} · 当前软件兼容版本 ${item.required}。请在上方Python环境中自行安装/切换该版本后重新检查。`);
        if (plan.missing_weights.length) rows.push('请自行将官方权重放入 '+plan.root+'/environment-v1/weights/，校验通过后再连接：'+plan.missing_weights.join('、'));
      }
      if(plan.optional_missing_weights?.length)rows.push('可选模式缺少权重：'+plan.optional_missing_weights.join('、')+'，对应模式暂不可用。');
      rows.push(...(plan.guidance||[]));
      $('#ssh-environment-checks').replaceChildren(...rows.map(row => {
        const item = document.createElement('li');
        item.textContent = row;
        return item;
      }));
      $('#ssh-prepare-description').textContent=plan.missing.length?`在所选环境手动安装兼容训练库。PyTorch/Torchvision需使用 ${plan.recipe} 对应的CUDA构建；安装后重新连接并检查。`:'';
      $('#ssh-prepare').hidden=phase==='ready'||Boolean(plan.blockers.length||plan.missing.length||plan.missing_weights.length);
      $('#ssh-enable-candidate').hidden = phase !== 'ready' || state.active === `ssh-${candidate.id}`;
      $('#ssh-status').textContent = {
        checked: '已连接，查看校验结果并处理缺失项。', preparing: '正在校验已有环境。', validating: '正在检查CUDA、NMS及模型前向，请稍候。',
        ready: '环境支持训练，可启用此目标。', failed: '环境校验未通过，请按提示自行处理后重新检查。', interrupted: '环境校验已中断。',
      }[phase] ?? '';
      issue(candidate.error || '');
    }
    clearTimeout(timer);
    if (dialog.open && ['preparing','validating'].includes(candidate?.state)) timer = setTimeout(refresh, 2500);
    sync();
  }

  async function refresh() {
    try { render(await request(endpoint)); }
    catch (error) { issue(error.message); }
  }

  async function action(operation, body, message) {
    if (busy) return;
    busy = true;
    issue('');
    $('#ssh-status').textContent = message;
    sync();
    try {
      render(await request(`${endpoint}/${operation}`, {method:'POST', body}));
      if (operation === 'activate') {
        const info = await request('/api/remote');
        dialog.close();
        await onSaved(info);
      }
      if (operation === 'connect') {
        $('#ssh-password').value = '';
        clearKey();
      }
    } catch (error) {
      issue(error.message);
      $('#ssh-status').textContent = '';
    } finally {
      busy = false;
      sync();
    }
  }

  function connectionKind(){
    const local=$('#ssh-kind').value==='local';form.hidden=local;$('#ssh-check-local').hidden=!local;
    if($('#ssh-kind').value==='local_setup'&&$('#ssh-command').value!=='ssh -p 22226 root@127.0.0.1'){clearKey();$('#ssh-command').value='ssh -p 22226 root@127.0.0.1';}
  }
  $('#ssh-kind').onchange=connectionKind;
  $('#ssh-check-local').onclick=()=>action('local',{python:$('#ssh-python').value.trim()||undefined,root:$('#ssh-workspace').value},'正在检查本机已有环境…');
  $('#ssh-command').addEventListener('input', clearKey);
  $('#ssh-confirm').onchange = sync;
  $('#ssh-instance-select').onchange = sync;
  $('#ssh-reveal').onclick = () => {
    const input = $('#ssh-password');
    const show = input.type === 'password';
    input.type = show ? 'text' : 'password';
    $('#ssh-reveal').textContent = show ? '隐藏密码' : '显示密码';
    $('#ssh-reveal').setAttribute('aria-pressed', String(show));
  };
  $('#ssh-read-key').onclick = async () => {
    if (busy || !$('#ssh-command').reportValidity()) return;
    busy = true;
    clearKey();
    issue('');
    try {
      key = await request('/api/remote/host-key', {method:'POST', body:{command:$('#ssh-command').value.trim()}});
      $('#ssh-fingerprint').textContent = `服务器公钥 ${key.fingerprint}`;
      $('#ssh-confirm-label').hidden = false;
    } catch (error) { issue(error.message); }
    finally { busy = false; sync(); }
  };
  form.onsubmit = event => {
    event.preventDefault();
    if (!key || !$('#ssh-confirm').checked) return;
    action('connect', {command:$('#ssh-command').value.trim(), password:$('#ssh-password').value, fingerprint:key.fingerprint, confirmed_host:true,connection_kind:$('#ssh-kind').value==='local_setup'?'local':'ssh',python:$('#ssh-python').value.trim()||undefined,root:$('#ssh-workspace').value}, '正在登录并检查 GPU、驱动及 Python 依赖…');
  };
  $('#ssh-prepare').onclick = () => action('validate', {id:candidate.id}, '正在校验已有环境…');
  $('#ssh-enable-candidate').onclick = () => action('activate', {id:`ssh-${candidate.id}`}, '正在复核环境并切换目标…');
  $('#ssh-activate').onclick = () => action('activate', {id:$('#ssh-instance-select').value}, '正在复核环境并切换目标…');
  $('#ssh-close').onclick = () => { if (!busy) dialog.close(); };
  dialog.addEventListener('cancel', event => { if (busy) event.preventDefault(); });
  dialog.addEventListener('close', () => {
    clearTimeout(timer);
    form.reset();
    $('#ssh-password').type = 'password';
    $('#ssh-reveal').textContent = '显示密码';
    $('#ssh-reveal').setAttribute('aria-pressed','false');
    clearKey();
  });
  return {open() { issue('');connectionKind();dialog.showModal();refresh(); }};
}
