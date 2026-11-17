import { request,uploadTrainingResult } from './api.js';
import { createModelWorkspace } from './models.js';
import { createTrainingReport } from './training-report.js';
import { createTrainingConfiguration } from './training-config.js';
import { createSSHConnection } from './ssh-connection.js';
const $=s=>document.querySelector(s);
const time=v=>new Date(v).toLocaleString('zh-CN',{hour12:false});
const label=n=>'V'+String(n).padStart(3,'0');
const states={QUEUED:'待执行',CANCELLED:'已取消',DISPATCHED:'已发出 · 等待回传',RUNNING:'实例进程运行中',SUCCEEDED:'训练完成',FAILED:'训练失败',INTERRUPTED:'已中断'};
const phases={UPLOADING:'上传执行包',STARTING:'启动实例进程',RUNNING:'运行中',WAITING_RESULT:'等待结果收尾',RECONNECTING:'恢复连接',DOWNLOADING:'下载并校验结果',COMPLETED:'结果已接收',ERROR:'连接中断，需处理',CANCELLING:'正在中止并核对资源',RELEASING:'结果已生成，核实进程和设备锁释放',PAUSED:'监控已暂停，实例计算继续'};

export function createTrainingWorkspace({notify}) {
  const models=createModelWorkspace();
  const report=createTrainingReport();
  const configEditor=createTrainingConfiguration({onChange:()=>controls()});
  const sshConnection=createSSHConnection({onSaved:async info=>{
    remoteInfo=info;
    connectionInfo();
    await reload();
    if (dialog.open && options) {
      const manual=$('#training-parameter-mode').value==='manual';
      const previous=configEditor.value();
      fillTargets();
      configEditor.reset(context.projectId,context.taskId,options,previous,manual);
      versionInfo();
    }
    notify('训练目标已切换');
  }});
  function connectionInfo(){const ssh=$('#training-execution-method').value==='ssh';$('#training-connection-settings').hidden=!ssh;$('#training-connection-info').textContent=remoteInfo?.configured?`已配置：${remoteInfo.endpoint} · ${remoteInfo.authentication==='password'?'密码登录':'免密登录'}`:'请先设置SSH登录指令和密码。';controls();}
  let context=null,epoch=0,options=null,versions=[],busy=false,draft=null,rows=[],loading=false,uploadContext=null,remoteInfo=null,viewedJob=null,view='active',visibilityDraft=null;
  const dialog=$('#training-dialog'),form=$('#training-form'),detailDialog=$('#training-detail-dialog');
  const base=c=>`/api/projects/${c.projectId}/tasks/${c.taskId}/training`;
  function error(message){for(const id of ['#training-error','#training-model-error']){$(id).textContent=message;$(id).hidden=!message;}}
  function controls(){
    $('#training-model-mode').disabled=Boolean(draft?.sourceModelId);
    if(draft?.sourceModelId)$('#training-weight-info').textContent=`起始模型：v${draft.sourceModelNumber} · best.pt`;
    $('#training-list').inert=loading||busy;$('#training-list').setAttribute('aria-busy',String(loading||busy));
    $('#training-new').disabled=loading || busy || !versions.length || !options;
    $('#training-refresh').disabled=loading || busy;
    $('#training-view').disabled=loading || busy;
    $('#training-clean').disabled=loading || busy || !rows.some(j=>view==='trash' || (j.can_delete && j.state!=='QUEUED'));
    $('#training-empty-version').hidden=loading || Boolean(versions.length);
    $('#training-submit').disabled=loading || busy || !configEditor.verified() || !options?.targets.some(t=>t.enabled!==false) || ($('#training-execution-method').value==='ssh' && (!remoteInfo?.configured || remoteInfo.execution_enabled===false));
    const readiness=$('#training-ready-status');readiness.hidden=loading || busy || !$('#training-submit').disabled;readiness.textContent=configEditor.pending()?'正在准备训练方案…':!remoteInfo?.configured?'请先在高级选项中连接训练设备。':'当前方案暂不可运行，请在高级选项中选择已验证的方案。';
  }
  function render(){
    $('#training-list').replaceChildren();$('#training-empty').hidden=Boolean(rows.length);$('#training-empty').textContent=view==='trash'?'回收站为空。':'暂无训练记录。';$('#training-clean').textContent=view==='trash'?'恢复回收站记录':'清理已结束记录';
    for(const job of rows){
      const card=document.createElement('article');card.className='training-row';
      const body=document.createElement('div'),name=document.createElement('h4'),info=document.createElement('p'),state=document.createElement('span');
      name.textContent=job.name;info.textContent=`${label(job.snapshot.data_version_number)} · ${job.snapshot.requested_parameters.epochs} 轮 · ${job.snapshot.target.name} · ${job.dispatch?.result?.report?.started_at?"开始："+time(job.dispatch.result.report.started_at):"创建："+time(job.created_at)}`;
      state.className='training-state';state.dataset.state=job.state;state.textContent=job.remote?.progress?.local_abandonment_resolution?'已结束 · 设备已重新启用':job.remote?.progress?.local_abandonment?'设备不可用 · 本地已结束':job.state==='QUEUED'?`待执行 · 队列第 ${job.queue_position} 位`:states[job.state];
      body.append(name,info,state);
      if(job.snapshot.initial_model){const origin=document.createElement('p');origin.textContent=`基于模型 v${job.snapshot.initial_model.number} 训练`;body.append(origin);}
      const seconds=job.dispatch?.result?.report?.elapsed_seconds;const duration=document.createElement('p');duration.className='training-duration';duration.textContent=Number.isFinite(seconds)?`训练执行总耗时：${Math.floor(seconds/3600)}小时 ${Math.floor(seconds%3600/60)}分 ${(seconds%60).toFixed(1)}秒`:'训练耗时：等待完成';duration.title='远端回传的执行耗时，包含环境检查、训练和权重核对；不含网页等待及文件传输。';body.append(duration);
      if(job.remote){const progress=document.createElement('p');progress.className='training-live';progress.textContent=`进度：${job.remote.progress.local_abandonment?'本地已结束跟踪':phases[job.remote.phase] || job.remote.phase} · 已完成 ${job.remote.progress.epochs_completed ?? 0}/${job.snapshot.requested_parameters.epochs} 轮 · 更新于 ${time(job.remote.updated_at)}`;body.append(progress);if(job.remote.error){const issue=document.createElement('p');issue.className='training-live-error';issue.textContent=job.remote.error;body.append(issue);}}
      const actions=document.createElement('div');actions.className='training-actions';
      const view=document.createElement('button');view.className='button secondary';view.textContent='查看训练详情';view.onclick=()=>showDetail(job);actions.append(view);
      if(job.state==='SUCCEEDED' && (!job.deleted_at || job.model_version))addModelAction(actions,job);
      if(job.deleted_at){
        const restore=document.createElement('button');restore.className='button secondary';restore.textContent='恢复记录';restore.onclick=()=>visibility([job],true);actions.append(restore);
        const more=document.createElement('details');more.className='row-more';const summary=document.createElement('summary');summary.textContent='更多操作';more.append(summary);
      for(const action of [...actions.children]){if(!action.textContent.startsWith('查看模型') && !['查看训练详情','开始训练','恢复连接','停止训练','取消等待','查看模型','查看模型版本'].includes(action.textContent))more.append(action);else if(action.textContent==='开始训练')action.className='button primary';}
      if(more.children.length>1)actions.append(more);
      card.append(body,actions);$('#training-list').append(card);continue;
      }
      if(job.state==='QUEUED'){
        const otherTarget=options?.targets[0]?.id!==job.snapshot.target.id;
        if(job.snapshot.execution_method!=='manual' && remoteInfo?.configured && remoteInfo.execution_enabled!==false)addRemoteAction(actions,job,'开始训练','remote-run',otherTarget || job.queue_position!==1);
        const dispatch=document.createElement('button');dispatch.className='button primary';dispatch.textContent=otherTarget?'请先切回此记录的目标':'生成执行包';dispatch.disabled=otherTarget || job.queue_position!==1 || remoteInfo?.execution_enabled===false;
        dispatch.onclick=async()=>{
          if(busy)return;busy=true;controls();dispatch.disabled=true;dispatch.textContent='核对并打包…';const target=context,token=epoch;
          try{const data=await request(`${base(target)}/${job.id}/dispatch`,{method:'POST',body:{}});if(token===epoch){await reload(target,token);execution(data.job);}}
          catch(e){if(token===epoch)error(e.message);}finally{busy=false;controls();if(token===epoch)render();}
        };actions.append(dispatch);
        const cancel=document.createElement('button');cancel.className='button secondary';cancel.textContent='取消等待';
        cancel.onclick=async()=>{
          if(busy)return;busy=true;controls();cancel.disabled=true;const target=context,token=epoch;
          try{await request(`${base(target)}/${job.id}/cancel`,{method:'POST',body:{}});if(token===epoch){notify('等待记录已取消');await reload(target,token);}}
          catch(e){if(token===epoch)error(e.message);}finally{busy=false;controls();}
        };actions.append(cancel);
      }else if(job.state==='DISPATCHED' || job.state==='RUNNING'){
        if(remoteInfo?.configured){addRemoteAction(actions,job,job.remote?'恢复连接':'开始训练','remote-run');if(job.remote){addRemoteAction(actions,job,'停止训练','remote-stop');if(job.state==='RUNNING' && job.remote.phase!=='PAUSED' && !job.remote.cancel_requested)addRemoteAction(actions,job,'暂停监控','remote-pause');}}
        const instructions=document.createElement('button');instructions.className='button secondary';instructions.textContent='下载与运行说明';instructions.onclick=()=>execution(job);
        const upload=document.createElement('button');upload.className='button primary';upload.textContent='回传结果';upload.onclick=()=>chooseResult(job);
        actions.append(instructions,upload);
      }else{
        const again=document.createElement('button');again.className='button secondary';again.textContent='按此配置新建';again.onclick=()=>edit(job);actions.append(again);
      }
      const remove=document.createElement('button');remove.className='button secondary danger-hover';remove.textContent=job.can_finish_stop?'核实中止并删除':job.can_abandon?'设备不可用，结束并删除':'删除记录';remove.disabled=!(job.can_delete || job.can_abandon || job.can_finish_stop);remove.title=job.can_finish_stop?'确认原进程退出及锁释放后移入回收站':job.can_abandon?'确认设备不可用后结束本地跟踪并移入回收站':job.can_delete?'移入回收站，保留权重和模型来源':'运行中或尚未收尾，请先中止并确认资源释放';remove.onclick=()=>visibility([job],false,job.can_abandon&&!job.can_finish_stop,job.can_finish_stop);actions.append(remove);
      if(job.can_finish_stop&&job.can_abandon){const unavailable=document.createElement('button');unavailable.className='button secondary';unavailable.textContent='设备确实不可用';unavailable.onclick=()=>visibility([job],false,true);actions.append(unavailable);}
      const more=document.createElement('details');more.className='row-more';const summary=document.createElement('summary');summary.textContent='更多操作';more.append(summary);
      for(const action of [...actions.children]){if(!action.textContent.startsWith('查看模型') && !['查看训练详情','开始训练','恢复连接','停止训练','取消等待','查看模型','查看模型版本'].includes(action.textContent))more.append(action);else if(action.textContent==='开始训练')action.className='button primary';}
      if(more.children.length>1)actions.append(more);
      card.append(body,actions);$('#training-list').append(card);
    }
  }
  function visibility(records,restore=false,abandon=false,finishStop=false){
    if(busy || !records.length)return;
    visibilityDraft={context:{...context},token:epoch,records:records.slice(0,200).map(j=>({id:j.id,confirmation_name:j.name,visibility_revision:j.visibility_revision})),restore,abandon,finishStop};
    $('#training-visibility-title').textContent=restore?'恢复训练记录':`删除 ${visibilityDraft.records.length} 条训练记录`;
    $('#training-visibility-impact').textContent=restore?'恢复到训练列表；已取消的等待任务仍保持取消，不会自动启动。':'记录将移入回收站，可恢复。等待中的任务会先取消；权重、日志、验证证据和已登记模型的来源均保留，本次不释放磁盘空间。';
    $('#training-visibility-names').textContent=visibilityDraft.records.map(j=>j.confirmation_name).join('\n');$('#training-visibility-error').hidden=true;
    if(finishStop)$('#training-visibility-impact').textContent='核对原训练进程已退出、任务锁和设备锁已释放后，将记录移入回收站。日志和权重保留，设备可继续用于后续训练。';
    $('#training-visibility-submit').textContent=restore?'确认恢复':'移入回收站';if(abandon){$('#training-visibility-title').textContent='确认设备不可用';$('#training-visibility-impact').textContent='确认该设备无法使用后，将结束本地跟踪、解除本地占用并移入回收站。远端停止状态仍未核实，该设备将禁止继续派发训练。已有文件保留，恢复记录不会重启任务。';$('#training-visibility-submit').textContent='确认设备不可用，结束并删除';}$('#training-visibility-dialog').showModal();
  }
  $('#training-visibility-form').onsubmit=async event=>{
    event.preventDefault();if(busy || !visibilityDraft)return;
    const current=visibilityDraft;busy=true;controls();$('#training-visibility-submit').disabled=true;$('#training-visibility-close').disabled=true;$('#training-visibility-error').hidden=true;
    try{
      if(current.finishStop){
        const record=current.records[0];await request(`${base(current.context)}/${record.id}/remote-stop-and-trash`,{method:'POST',body:{confirmation_name:record.confirmation_name,visibility_revision:record.visibility_revision}});
        if(current.token===epoch){$('#training-visibility-dialog').close();notify('已核实中止并移入回收站，设备可继续使用');await reload(current.context,current.token);}return;
      }
      const result=current.abandon?await request(`${base(current.context)}/${current.records[0].id}/remote-abandon`,{method:'POST',body:{confirmation_name:current.records[0].confirmation_name,visibility_revision:current.records[0].visibility_revision,instance_unavailable:true}}):await request(`${base(current.context)}/${current.restore?'restore':'trash'}`,{method:'POST',body:{records:current.records}});
      if(current.token!==epoch)return;$('#training-visibility-dialog').close();notify(current.abandon?'已结束本地跟踪并移入回收站，请选择其他已验证实例':current.restore?`已恢复 ${result.changed} 条记录`:`已移入回收站 ${result.changed} 条记录`);await reload(current.context,current.token);
    }catch(e){if(current.token===epoch){$('#training-visibility-error').textContent=e.message;$('#training-visibility-error').hidden=false;}}
    finally{busy=false;$('#training-visibility-submit').disabled=false;$('#training-visibility-close').disabled=false;controls();}
  };
  $('#training-visibility-close').onclick=()=>{if(!busy)$('#training-visibility-dialog').close();};
  $('#training-visibility-dialog').addEventListener('cancel',event=>{if(busy)event.preventDefault();});
  function addModelAction(container,job){
    const button=document.createElement('button');button.className='button primary';
    button.textContent=job.model_version?`查看模型 v${job.model_version.number}`:'模型文件准备中';button.disabled=!job.model_version;
    button.onclick=()=>models.view(job.model_version.id);container.append(button);
  }
  function addRemoteAction(actions,job,title,action,disabled=false){
    const button=document.createElement('button');button.className='button '+(action==='remote-run'?'primary':'secondary');button.textContent=title;button.disabled=disabled;
    button.onclick=async()=>{
      if(busy)return;busy=true;controls();button.disabled=true;const target=context,token=epoch;error('');
      try{await request(`${base(target)}/${job.id}/${action}`,{method:'POST',body:{}});if(token===epoch){notify(action==='remote-run'?'SSH 操作已提交，页面将自动更新':action==='remote-pause'?'监控已暂停；实例计算继续，恢复连接不会重跑':'已请求中止，确认进程结束并回收结果后才释放目标');await reload(target,token);}}
      catch(e){if(token===epoch)error(e.message);}finally{busy=false;controls();}
    };actions.append(button);
  }
  async function reload(target=context,token=epoch){
    loading=true;controls();error('');
    try{
      const [config,data,list,connection]=await Promise.all([request(`/api/training-options?project_id=${target.projectId}&task_id=${target.taskId}`),request(`/api/projects/${target.projectId}/tasks/${target.taskId}/versions`),request(base(target)+`?view=${view}`),request('/api/remote')]);
      if(token!==epoch)return;options=config;versions=data.versions;rows=list.jobs;remoteInfo=connection;render();
      await models.refresh();
      $('#training-device-label').textContent=connection.configured?`训练设备：${options?.targets?.[0]?.name || '已连接设备'}`:'尚未连接训练设备，请展开下方连接设置。';
      $('#training-ssh-status').textContent=connection.configured?`SSH 目标：${connection.endpoint} · ${connection.last_check?`上次连接检查 ${time(connection.last_check.checked_at)}`:'等待连接检查'}`:'SSH尚未配置，可使用手动执行包。';
      if(connection.pending_target)$('#training-ssh-status').textContent=`新目标：${connection.pending_target.endpoint} · ${connection.pending_target.message}。旧连接保留，未向旧实例发送新任务。`;
      if(!connection.pending_target && connection.configured){const active=connection.reservations?.length || 0;$('#training-ssh-status').textContent+=` · ${active?`当前有 ${active} 个活动训练`:'当前没有活动训练'}`;}
      $('#training-ssh-check').disabled=!connection.configured || connection.execution_enabled===false;
      $('#training-device-recover').hidden=!options.targets.some(target=>target.unavailable);
      if(detailDialog.open && viewedJob){const row=rows.find(j=>j.id===viewedJob);if(row)showDetail(row);}
    }catch(e){if(token===epoch)error(e.message);}
    finally{if(token===epoch){loading=false;controls();}}
  }
  async function open(projectId,taskId){
    context={projectId,taskId};options=null;versions=[];rows=[];draft=null;view='active';$('#training-view').value=view;visibilityDraft=null;if($('#training-visibility-dialog').open)$('#training-visibility-dialog').close();
    if(dialog.open)dialog.close();if(detailDialog.open)detailDialog.close();if($('#training-execution-dialog').open)$('#training-execution-dialog').close();render();++epoch;await models.open(projectId,taskId);await reload(context,epoch);
  }
  function versionInfo(){
    configEditor.changedVersion();
    const version=versions.find(v=>v.id===$('#training-version').value);
    $('#training-version-info').textContent=version?`${version.image_count} 张图片 · ${version.class_count} 类 · 训练 ${version.distribution.train.image_count} / 验证 ${version.distribution.val.image_count}；仅使用此冻结版本。`:'';
  }
  function fillTargets(){
    $('#training-target').replaceChildren();
    for(const item of [...options.targets,...options.future_targets]){
      const option=new Option(item.name+(item.enabled===false?` · ${item.reason}`:' · 已验证目标'),item.id);
      option.disabled=item.enabled===false;
      $('#training-target').append(option);
    }
  }
  function edit(previous=null,source=null){
    if(busy || !options || !versions.length)return;
    draft={context:{...context},token:epoch,key:crypto.randomUUID().replaceAll('-',''),payload:null,sourceModelId:source?.id || previous?.snapshot.initial_model?.id,sourceModelNumber:source?.number || previous?.snapshot.initial_model?.number};
    let origin=$('#training-source-model');if(!origin){origin=document.createElement('p');origin.id='training-source-model';form.prepend(origin);}origin.hidden=!draft.sourceModelId;origin.textContent=draft.sourceModelId?`从模型 v${source?.number || previous.snapshot.initial_model.number} 的 best.pt 开始训练。请选择补标后的数据版本。`:'';
    form.reset();$('#training-form-error').hidden=true;$('#training-model-mode').value=previous?.snapshot.training_mode || 'standard';$('#training-execution-method').value=previous?.snapshot.execution_method || 'ssh';connectionInfo();$('#training-weight-info').textContent=`预训练模型：${options.task_type==='segment'?'SegDL':'DetectDL'}`;
    for(const node of form.querySelectorAll('[data-error]'))node.textContent='';
    $('#training-version').replaceChildren();
    for(const version of versions){const option=document.createElement('option');option.value=version.id;option.textContent=`${label(version.number)} · ${version.image_count} 张 · ${version.class_count} 类`;$('#training-version').append(option);}
    if(previous && versions.some(v=>v.id===previous.version_id))$('#training-version').value=previous.version_id;
    $('#training-name').value=previous?`${previous.name.slice(0,55)}（新建）`:`训练 ${label(versions[0].number)}`;
    $('#training-epochs').value=previous?.snapshot.requested_parameters.epochs || options.default_epochs;
    fillTargets();
    const priorConfiguration=previous?.snapshot.configuration;
    const configuration=source&&!options.configuration_profiles.some(p=>p.id===priorConfiguration?.profile&&p.enabled)?options.default_configuration:priorConfiguration;
    configEditor.reset(context.projectId,context.taskId,options,configuration,Boolean(previous));versionInfo();dialog.showModal();$('#training-name').focus();
  }
  function showDetail(job){
    viewedJob=job.id;
    $('#training-detail-title').textContent=`${job.name} · ${job.snapshot.requested_parameters.epochs} 轮`; 
    $('#training-detail-state').textContent=`${states[job.state]} · ${job.execution_message}`;
    const s=job.snapshot,p=s.requested_parameters;
    if(s.initial_model)$('#training-detail-state').textContent+=` · 基于模型 v${s.initial_model.number}`;
    $('#training-detail-summary').textContent=`数据：${label(s.data_version_number)} · ${s.image_count} 张 · ${s.class_count} 类\n训练 / 验证：${s.distribution.train.image_count} / ${s.distribution.val.image_count} 张\n设备：${s.target.name}\n模式：${s.training_mode==='high_accuracy'?'高精度':'通用'} · ${p.epochs} 轮\n创建时间：${time(job.created_at)}`;

    $('#training-detail-snapshot').textContent=JSON.stringify(job.snapshot,null,2);
    $('#training-detail-hash').textContent=`配置 SHA-256：${job.snapshot_sha256}`;
    $('#training-snapshot-download').href=`${base(context)}/${job.id}/snapshot`;
    const links=$('#training-artifact-links');links.replaceChildren();const receipt=job.dispatch?.result;
    $('#training-detail-summary').replaceChildren(...$('#training-detail-summary').textContent.split('\n').map(line=>{const row=document.createElement('div');row.className='parameter-pair';const split=line.indexOf('：');const key=document.createElement('dt'),value=document.createElement('dd');key.textContent=split<0?'说明':line.slice(0,split);value.textContent=split<0?line:line.slice(split+1);row.append(key,value);return row;}));
    $('#training-result-summary').textContent=receipt?`${receipt.report.epochs_completed ?? '未知'} 轮已完成${receipt.report.completion_reason==='early_stopped'?' · 提前停止':''}。${receipt.report.error || (job.model_version?'模型已生成，可查看模型效果。':'正在准备模型文件。')}`:'';
    const failure=receipt?.report?.diagnostic || (!receipt && job.remote?.progress?.diagnostic);
    const connectionProblem=!receipt && job.remote?.error;
    $('#training-diagnostic').hidden=!failure && !connectionProblem && job.state!=='FAILED';
    $('#training-diagnostic-title').textContent=failure?.title || (connectionProblem?'连接或结果收尾需处理；训练状态保持原记录':'执行失败；历史报告未提供结构化分类');
    $('#training-diagnostic-action').textContent=failure?.next_step || (connectionProblem?(job.remote.progress.recovery_hint || job.remote.error):'查看原始日志定位原因；处理后按配置新建，原记录不会自动重跑。');
    $('#training-diagnostic-code').textContent=`分类：${failure?.code || job.remote?.progress?.error_code || 'LEGACY_FAILURE'}`;
    if(job.state==='SUCCEEDED' && (!job.deleted_at || job.model_version))addModelAction(links,job);
    if(receipt)for(const [name,title] of [['console.log','下载运行日志'],['results.csv','下载训练指标'],['args.yaml','下载实际参数'],['best.pt','下载 best.pt'],['last.pt','下载 last.pt']]){
      if(!receipt.artifacts[name])continue;const link=document.createElement('a');link.className='button secondary';link.href=`${base(context)}/${job.id}/artifacts/${name}`;link.download='';link.textContent=title;links.append(link);
    }
    const downloads=document.createElement('details');downloads.className='row-more';const caption=document.createElement('summary');caption.textContent='下载训练材料';downloads.append(caption);for(const link of [...links.querySelectorAll('a')])downloads.append(link);if(downloads.children.length>1)links.append(downloads);
    $('#training-live-log').textContent=(job.remote?.progress?.log_tail || '').replace(/\u001b\[[0-?]*[ -/]*[@-~]/g,'').replaceAll('\r','\n');
    $('#training-live-log-wrapper').hidden=!job.remote;
    report.open(context.projectId,context.taskId,job);
    if(!detailDialog.open)detailDialog.showModal();
  }
  function execution(job){
    const folder=`train-${job.id}`;
    $('#training-execution-title').textContent=`执行：${job.name}`;
    $('#training-package-download').href=`${base(context)}/${job.id}/package`;
    const python=job.snapshot.target.python_executable || 'python';
    const directory=job.snapshot.target.workspace_root || '/root/autodl-tmp';
    $('#training-execution-command').textContent=`cd ${directory}\nunzip -n ${folder}.zip -d ${folder}\ncd ${folder}\n${python} -u run_training_job.py`;
    $('#training-package-hash').textContent=`执行包 SHA-256：${job.dispatch.package_sha256}`;
    $('#training-execution-return').textContent=`执行结束后，下载此目录的 return-result.zip，回工作台点击本记录的“回传结果”。本次要求 ${job.snapshot.requested_parameters.epochs} 轮，终端会直接显示进度，并保存日志。`;
    $('#training-execution-recovery').textContent=`若未执行而决定取消，或进程被强杀后没有结果，在同目录运行 ${python} -u run_training_job.py --recover-interrupted。脚本确认设备和任务锁均空闲后生成中断结果，再回传到工作台。请保留锁文件和 started.json。`;
    $('#training-execution-dialog').showModal();
  }
  function chooseResult(job){
    if(busy)return;uploadContext={jobId:job.id,context:{...context},token:epoch};$('#training-result-file').value='';$('#training-result-file').click();
  }
  $('#training-result-file').onchange=async()=>{
    const file=$('#training-result-file').files[0],target=uploadContext;if(!file || !target || busy)return;
    if(file.size>128*1024**2){error('结果ZIP最多128MiB。');return;}
    busy=true;controls();error('');$('#training-upload-status').textContent='正在上传并核对结果，请保留本地ZIP…';
    try{const result=await uploadTrainingResult(`${base(target.context)}/${target.jobId}/result`,file);if(target.token===epoch){notify(result.reused?'该结果已接收，无需重复回传':'训练结果已接收并完成文件校验');await reload(target.context,target.token);showDetail(result.job);}}
    catch(e){if(target.token===epoch)error(e.message);}
    finally{busy=false;$('#training-upload-status').textContent='';controls();}
  };
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(busy || !draft)return;
    const body={name:$('#training-name').value.trim(),version_id:$('#training-version').value,target_id:$('#training-target').value,epochs:Number($('#training-epochs').value),training_mode:$('#training-model-mode').value,execution_method:$('#training-execution-method').value};
    if(draft.sourceModelId)body.source_model_id=draft.sourceModelId;
    const configuration=configEditor.value();if(configuration)body.configuration=configuration;
    // Retain the same key on a lost response; editing creates a new submission.
    const fingerprint=JSON.stringify(body);
    if(draft.payload && draft.payload!==fingerprint)draft.key=crypto.randomUUID().replaceAll('-','');
    draft.payload=fingerprint;body.idempotency_key=draft.key;
    const current=draft;busy=true;controls();$('#training-form-error').hidden=true;
    for(const node of form.querySelectorAll('[data-error]'))node.textContent='';
    for(const node of form.querySelectorAll('input,select,button'))node.disabled=true;
    $('#training-submit').textContent='核对数据并保存…';
    try{
      const result=await request(base(current.context),{method:'POST',body});
      if(current.token!==epoch)return;dialog.close();view=result.job.deleted_at?'trash':'active';$('#training-view').value=view;notify(result.reused?'已恢复上次提交，未重复创建':'训练配置已加入待执行队列');await reload(current.context,current.token);
    }catch(e){
      if(current.token!==epoch)return;$('#training-form-error').textContent=e.message;$('#training-form-error').hidden=false;
      for(const node of form.querySelectorAll('[data-error]'))node.textContent=e.fields?.[node.dataset.error] || '';
    }finally{
      busy=false;for(const node of form.querySelectorAll('input,select,button'))node.disabled=false;
      $('#training-submit').textContent='创建训练';configEditor.sync();controls();
    }
  });
  $('#training-execution-method').onchange=connectionInfo;$('#training-configure-ssh').onclick=$('#training-ssh-settings').onclick=()=>sshConnection.open(remoteInfo);
  $('#training-version').onchange=versionInfo;
  $('#training-new').onclick=()=>edit();$('#training-refresh').onclick=()=>{if(context && !busy)reload();};
  document.addEventListener('training-from-model',async event=>{
    if(!context || busy)return;const token=epoch;
    try{const {model}=await request(`/api/projects/${context.projectId}/tasks/${context.taskId}/models/${event.detail}`);if(token!==epoch)return;
      edit({name:`基于模型 v${model.number} 训练`,snapshot:model.manifest.training_snapshot},model);
    }catch(e){error(e.message);}
  });
  $('#training-view').onchange=()=>{if(!busy){view=$('#training-view').value;if(detailDialog.open)detailDialog.close();reload();}};
  $('#training-clean').onclick=()=>visibility(rows.filter(j=>view==='trash' || (j.can_delete && j.state!=='QUEUED')),view==='trash');
  $('#training-close').onclick=()=>{if(!busy)dialog.close();};
  dialog.addEventListener('cancel',event=>{if(busy)event.preventDefault();});
  dialog.addEventListener('close',()=>configEditor.invalidate());
  $('#training-detail-close').onclick=()=>detailDialog.close();
  $('#training-detail-top-close').onclick=()=>detailDialog.close();
  detailDialog.addEventListener('close',()=>report.close());
  $('#training-execution-close').onclick=()=>$('#training-execution-dialog').close();
  $('#training-ssh-check').onclick=async()=>{
    if(busy)return;busy=true;$('#training-ssh-check').disabled=true;
    try{const data=await request('/api/remote/check',{method:'POST',body:{}});notify(`SSH已连接：${data.python} · ${data.nvidia_smi}`);await reload();}catch(e){error(e.message);}finally{busy=false;$('#training-ssh-check').disabled=!remoteInfo?.configured || remoteInfo.execution_enabled===false;}
  };
  $('#training-device-recover').onclick=async()=>{
    if(busy)return;busy=true;$('#training-device-recover').disabled=true;
    try{await request('/api/remote/recover',{method:'POST',body:{}});notify('设备已重新核对，可继续训练');await reload();}catch(e){error(e.message);}finally{busy=false;$('#training-device-recover').disabled=false;controls();}
  };
  setInterval(()=>{if(context && !busy && !loading && $('#training-list').getClientRects().length && rows.some(j=>j.remote && ['DISPATCHED','RUNNING'].includes(j.state)))reload();},4000);
  return {open,refresh:()=>context?reload():Promise.resolve()};
}
