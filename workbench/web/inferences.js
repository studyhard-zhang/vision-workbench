import { request } from './api.js';
import { createInferenceStudio } from './inference-studio.js';
const $=s=>document.querySelector(s);
const states={QUEUED:'待执行',CANCELLED:'已取消',DISPATCHED:'已派发 · 目标占用中',RUNNING:'推理运行中',SUCCEEDED:'本批次处理完成',FAILED:'推理失败',INTERRUPTED:'已中断'};
const mid=n=>'v'+String(n);
function button(label,action,disabled=false){const b=document.createElement('button');b.type='button';b.className='button secondary';b.textContent=label;b.disabled=disabled;b.onclick=action;return b;}
function cell(row,value){const td=document.createElement('td');td.textContent=value;row.append(td);}

export function createInferenceWorkspace(){
  const studio=createInferenceStudio({onChanged:()=>refresh()});
  let context=null,epoch=0,rows=[],draft=null,busy=false,loading=false,confirmation=null,report=null,viewed=null,imageEpoch=0,detailEpoch=0;
  const createDialog=$('#inference-create-dialog'),form=$('#inference-create-form'),detail=$('#inference-detail-dialog'),confirmDialog=$('#inference-confirm-dialog');
  const base=c=>`/api/projects/${c.projectId}/tasks/${c.taskId}/inferences`;
  const inputBase=c=>`/api/projects/${c.projectId}/tasks/${c.taskId}/inference-inputs`;
  function error(text){$('#inference-error').textContent=text;$('#inference-error').hidden=!text;}
  function controls(){
    $('#inference-refresh').disabled=busy || loading;
    $('#inference-create-close').disabled=busy;$('#inference-files').disabled=busy;
    $('#inference-create-submit').disabled=busy || !draft || ![...draft.inputs.values()].some(i=>i.status==='READY') || !!draft.uploadErrors.length;
    $('#inference-confirm-submit').disabled=busy;$('#inference-confirm-close').disabled=busy;
    for(const b of document.querySelectorAll('#inference-input-history button,#inference-selected-inputs button'))b.disabled=busy;
  }
  function render(){
    const list=$('#inference-list');list.replaceChildren();$('#inference-empty').hidden=!!rows.length;
    for(const job of rows){
      const article=document.createElement('article');article.className='training-row';const body=document.createElement('div'),h=document.createElement('h4'),p=document.createElement('p'),actions=document.createElement('div');actions.className='training-row-actions';h.textContent=job.name;
      p.textContent=`${mid(job.snapshot.model_number)} · ${job.snapshot.sample_count} 个输入 / ${job.snapshot.valid_count} 张有效图 · ${states[job.state]}${job.remote?' · SSH '+job.remote.phase:''}`;body.append(h,p);
      actions.append(button('在工作区查看',()=>{studio.load(job.id);document.querySelector('#inference-studio').scrollIntoView({block:'start'});},busy),button('查看技术详情',()=>view(job.id),busy));
      if(job.state==='QUEUED')actions.append(button('SSH运行推理',()=>confirm(job,'remote-run'),busy),button('取消等待',()=>operate(job,'cancel'),busy));
      if(['DISPATCHED','RUNNING'].includes(job.state))actions.append(button('恢复推理连接',()=>confirm(job,'remote-run'),busy),button('中止推理',()=>confirm(job,'remote-stop'),busy));
      if(['SUCCEEDED','FAILED','INTERRUPTED','CANCELLED'].includes(job.state))actions.append(button('按配置新建推理',()=>create(job.model_id,job),busy));
      article.append(body,actions);list.append(article);
    }
    studio.updateRows(rows);controls();
  }
  async function refresh(){
    if(!context || loading)return;const token=epoch,target=context;loading=true;controls();
    try{const data=await request(base(target));if(token===epoch){rows=data.jobs;render();}}
    catch(e){if(token===epoch)error(e.message);}
    finally{if(token===epoch){loading=false;controls();}}
  }
  function renderInputs(){
    if(!draft)return;const list=$('#inference-selected-inputs');list.replaceChildren();const inputs=[...draft.inputs.values()];
    $('#inference-input-summary').textContent=`已选 ${inputs.length} 个输入 · 有效 ${inputs.filter(i=>i.status==='READY').length} 张 · 坏图 ${inputs.filter(i=>i.status==='INVALID').length} 个。坏图保留失败原因，不送入GPU。`;
    for(const input of inputs){const row=document.createElement('li'),text=document.createElement('span');text.textContent=`${input.filename} · ${input.status==='READY'?`${input.width} × ${input.height}（校正方向后）`:input.error}`;row.append(text,button('移出本次选择',()=>{draft.inputs.delete(input.id);renderInputs();},busy));list.append(row);}
    $('#inference-upload-errors').textContent=draft.uploadErrors.join('\n');$('#inference-dismiss-upload-errors').hidden=!draft.uploadErrors.length;controls();
  }
  async function create(modelId,prior=null){return studio.chooseModel(modelId,prior);}
  async function legacyCreate(modelId,prior=null){
    if(!context || busy)return;const token=epoch,target=context;error('');
    try{
      const options=await request(`${base(target)}/options?model_id=${modelId}`);if(token!==epoch)return;
      if(!options.model.available)throw new Error('模型文件完整性不通过。');
      draft={modelId,target,token,key:crypto.randomUUID().replaceAll('-',''),inputs:new Map((prior?.snapshot.inputs || []).map(i=>[i.id,i])),uploadErrors:[]};
      $('#inference-create-title').textContent=`图片推理 ${mid(options.model.number)}`;
      $('#inference-create-source').textContent=`best.pt SHA：${options.model.manifest.artifacts['best.pt'].sha256}\n类别：${options.model.manifest.classes.map(c=>`${c.index}: ${c.name}`).join('、')}\n计算目标：${options.target.name}`;
      $('#inference-size-note').textContent=`沿用模型训练输入尺寸 ${options.defaults.imgsz} / batch1 / FP32，不增强。`;$('#inference-name').value=`图片推理 ${mid(options.model.number)}`;$('#inference-base-conf').value=prior?.snapshot.requested_parameters.conf ?? options.defaults.conf;$('#inference-iou').value=prior?.snapshot.requested_parameters.iou ?? options.defaults.iou;
      $('#inference-files').value='';$('#inference-create-error').textContent='';$('#inference-upload-status').textContent='';
      const history=$('#inference-input-history');history.replaceChildren();
      for(const input of options.inputs){const row=document.createElement('div');row.className='inference-history-row';const text=document.createElement('span');text.textContent=`${input.filename} · ${input.status==='READY'?'有效图片':input.error}`;
        row.append(text,button('加入本次选择',()=>{if(draft.inputs.size>=50 && !draft.inputs.has(input.id)){$('#inference-create-error').textContent='每批最多50个输入。';return;}draft.inputs.set(input.id,input);renderInputs();},busy));history.append(row);}
      renderInputs();createDialog.showModal();
    }catch(e){if(token===epoch)error(e.message);}
  }
  $('#inference-files').onchange=async event=>{
    if(!draft || busy)return;const current=draft,files=[...event.target.files];if(current.inputs.size+files.length>50){$('#inference-create-error').textContent='每批最多50个输入，请减少本次选择。';return;}
    busy=true;controls();$('#inference-create-error').textContent='';
    for(let idx=0;idx<files.length;idx++){
      if(current.token!==epoch)break;const file=files[idx];$('#inference-upload-status').textContent=`正在保存 ${idx+1}/${files.length}：${file.name}`;
      try{
        const response=await fetch(inputBase(current.target),{method:'POST',headers:{'Content-Type':'application/octet-stream','X-File-Name':encodeURIComponent(file.name)},body:file});
        const data=await response.json();if(!response.ok)throw new Error(data.error?.message || '输入未保存。');
        if(current.token===epoch)current.inputs.set(data.input.id,data.input);
      }catch(e){if(current.token===epoch)current.uploadErrors.push(`${file.name}：${e.message}`);}
    }
    busy=false;if(current.token===epoch){$('#inference-upload-status').textContent='本次选择已处理；上传不会调用GPU。';renderInputs();}controls();
  };
  $('#inference-dismiss-upload-errors').onclick=()=>{if(draft && !busy){draft.uploadErrors=[];renderInputs();}};
  form.onsubmit=async event=>{
    event.preventDefault();if(!draft || busy || draft.uploadErrors.length || ![...draft.inputs.values()].some(i=>i.status==='READY') || !form.reportValidity())return;const current=draft;busy=true;controls();
    try{
      await request(base(current.target),{method:'POST',body:{name:$('#inference-name').value.trim(),model_id:current.modelId,input_ids:[...current.inputs.keys()],conf:Number($('#inference-base-conf').value),iou:Number($('#inference-iou').value),idempotency_key:current.key}});
      if(current.token===epoch){draft=null;createDialog.close();await refresh();}
    }catch(e){if(current.token===epoch)$('#inference-create-error').textContent=e.message;}
    finally{busy=false;controls();render();}
  };
  $('#inference-create-close').onclick=()=>createDialog.close();createDialog.addEventListener('cancel',e=>{if(busy)e.preventDefault();});
  function confirm(job,action){
    if(busy)return;confirmation={job,action,target:context,token:epoch};const s=job.snapshot;
    $('#inference-confirm-title').textContent=action==='remote-stop'?'中止原推理进程':'运行或恢复图片推理';
    $('#inference-confirm-text').textContent=`${job.name}\n模型 ${mid(s.model_number)} / best.pt\n${s.valid_count} 张有效图 · ${s.sample_count-s.valid_count} 个坏图\n${s.target.name} · ${s.requested_parameters.imgsz} / batch1 / FP32，不增强\n最低置信度 ${s.requested_parameters.conf}，NMS IoU ${s.requested_parameters.iou}\n${action==='remote-stop'?'停止并回收原任务；核实进程和设备锁释放后才放行。':'首次运行使用GPU逐图推理，不训练或评估；恢复只核对原进程与结果，不重复推理。'}`;
    $('#inference-confirm-error').textContent='';controls();confirmDialog.showModal();
  }
  async function operate(job,action){
    if(busy)return;const token=epoch,target=context;busy=true;controls();error('');
    try{await request(`${base(target)}/${job.id}/${action}`,{method:'POST',body:{}});if(token===epoch){confirmDialog.close();await refresh();}}
    catch(e){if(token===epoch){error(e.message);$('#inference-confirm-error').textContent=e.message;}}
    finally{busy=false;controls();if(token===epoch)render();}
  }
  $('#inference-confirm-submit').onclick=()=>{if(confirmation?.token===epoch)operate(confirmation.job,confirmation.action);};$('#inference-confirm-close').onclick=()=>confirmDialog.close();confirmDialog.addEventListener('cancel',e=>{if(busy)e.preventDefault();});

  async function draw(){
    const data=report,target=context,id=viewed,ticket=++imageEpoch;if(!data)return;const row=data.images.find(i=>i.id===$('#inference-image').value);if(!row)return;
    const conf=Number($('#inference-display-conf').value),minimum=data.snapshot.requested_parameters.conf;
    const canvas=$('#inference-canvas'),body=$('#inference-boxes');body.replaceChildren();canvas.hidden=true;$('#inference-overlay-download').hidden=true;
    if(!Number.isFinite(conf) || conf<minimum || conf>1){$('#inference-image-status').textContent=`显示阈值需在 ${minimum} 至1之间。`;$('#inference-filtered-json').hidden=true;return;}
    $('#inference-filtered-json').hidden=false;$('#inference-filtered-json').href=`${base(target)}/${id}/export?conf=${conf}`;
    if(row.status!=='SUCCEEDED'){$('#inference-image-status').textContent=`${row.filename} · ${row.status==='NOT_PROCESSED'?'未处理':'失败'}：${row.error}`;return;}
    const boxes=row.predictions.filter(b=>b.confidence>=conf);
    $('#inference-image-status').textContent=`${row.filename} · 成功${boxes.length?` · ${boxes.length} 个目标`:' · 当前阈值下无检测目标'} · 校正方向后 ${row.width} × ${row.height}。源文件像素尺寸 ${row.source_width ?? row.width} × ${row.source_height ?? row.height}。`;
    for(const b of boxes){const tr=document.createElement('tr');for(const value of [b.class_index,b.class_name,b.confidence.toFixed(4),b.xyxy.map(v=>v.toFixed(2)).join(', ')])cell(tr,value);body.append(tr);}
    $('#inference-overlay-download').hidden=false;$('#inference-overlay-download').href=`${base(target)}/${id}/images/${row.id}/overlay?conf=${conf}`;
    const image=new Image();image.src=`${base(target)}/${id}/images/${row.id}/view`;
    try{await image.decode();if(ticket!==imageEpoch || report!==data)return;const scale=Math.min(1,1000/row.width);canvas.width=Math.round(row.width*scale);canvas.height=Math.round(row.height*scale);const c=canvas.getContext('2d');c.drawImage(image,0,0,canvas.width,canvas.height);c.lineWidth=2;c.font='13px sans-serif';
      if($('#inference-show-boxes').checked)for(const b of boxes){const [x1,y1,x2,y2]=b.xyxy;c.strokeStyle='#e76c17';c.fillStyle='#e76c17';if(b.segments){c.beginPath();for(const ring of b.segments){ring.points.forEach(([x,y],i)=>i?c.lineTo(x*scale,y*scale):c.moveTo(x*scale,y*scale));c.closePath();}c.fillStyle='#e76c1744';c.fill('evenodd');c.stroke();c.fillStyle='#e76c17';}else c.strokeRect(x1*scale,y1*scale,(x2-x1)*scale,(y2-y1)*scale);c.fillText(`${b.class_name} ${b.confidence.toFixed(3)}`,x1*scale+2,Math.max(14,y1*scale-3));}canvas.hidden=false;
    }catch{if(ticket===imageEpoch)$('#inference-image-status').textContent='输入图片加载失败，请核对原文件与完整性。';}
  }
  async function view(id){
    if(!context)return;const token=epoch,target=context,ticket=++detailEpoch;viewed=id;error('');
    try{
      const {job}=await request(`${base(target)}/${id}`);if(token!==epoch || ticket!==detailEpoch)return;report=null;imageEpoch++;
      $('#inference-detail-title').textContent=job.name;$('#inference-detail-status').textContent=`${states[job.state]}${job.remote?' · SSH '+job.remote.phase:''}`;
      const s=job.snapshot;$('#inference-detail-source').textContent=`模型 ${mid(s.model_number)} · ${s.model_version_id}\nbest.pt SHA ${s.weights_sha256}\n${s.sample_count} 个输入 / ${s.valid_count} 张有效图 · ${s.requested_parameters.imgsz} / batch1 / FP32\n最低置信度 ${s.requested_parameters.conf} · NMS IoU ${s.requested_parameters.iou}\n坐标以EXIF校正方向后的像素图片为准。`;
      const failure=job.dispatch?.result?.report;$('#inference-detail-error').textContent=[job.remote?.error || failure?.error,failure?.diagnostic?.title,failure?.diagnostic?.next_step].filter(Boolean).join('\n');
      $('#inference-progress').textContent=JSON.stringify(job.remote?.progress || {},null,2);$('#inference-snapshot').textContent=JSON.stringify(s,null,2);$('#inference-report-panel').hidden=true;$('#inference-downloads').replaceChildren();
      for(const n of Object.keys(job.dispatch?.result?.artifacts || {})){const a=document.createElement('a');a.className='button secondary';a.textContent=`下载原始 ${n}`;a.download='';a.href=`${base(target)}/${id}/artifacts/${n}`;$('#inference-downloads').append(a);}
      if(job.dispatch?.result?.artifacts?.['report.json']){
        const data=await request(`${base(target)}/${id}/report`);if(token!==epoch || ticket!==detailEpoch)return;report=data;$('#inference-report-panel').hidden=false;
        const counts={SUCCEEDED:0,FAILED:0,NOT_PROCESSED:0};for(const row of data.images)counts[row.status]++;
        $('#inference-result-summary').textContent=`逐图结果：${counts.SUCCEEDED} 张成功 · ${counts.FAILED} 张失败 · ${counts.NOT_PROCESSED} 张未处理。无目标属于成功。${data.complete?'':'这是未完成任务的部分结果。'}`;
        const select=$('#inference-image');select.replaceChildren();for(const row of data.images){const o=document.createElement('option');o.value=row.id;o.textContent=`${row.filename} · ${row.status==='SUCCEEDED'?'成功':row.status==='FAILED'?'失败':'未处理'}`;select.append(o);}
        $('#inference-display-conf').min=s.requested_parameters.conf;$('#inference-display-conf').value=Math.max(0.25,s.requested_parameters.conf);$('#inference-timing').textContent=JSON.stringify(data.timing,null,2);await draw();
      }
      if(token===epoch && ticket===detailEpoch && !detail.open)detail.showModal();
    }catch(e){if(token===epoch && ticket===detailEpoch)error(e.message);}
  }
  $('#inference-image').onchange=draw;$('#inference-display-conf').oninput=draw;$('#inference-show-boxes').onchange=draw;
  $('#inference-detail-close').onclick=()=>{detail.close();detailEpoch++;imageEpoch++;};$('#inference-detail-refresh').onclick=()=>view(viewed);$('#inference-refresh').onclick=refresh;
  setInterval(()=>{if(context && !busy && !loading && $('#inference-list').getClientRects().length && rows.some(j=>['DISPATCHED','RUNNING'].includes(j.state)))refresh();},4000);
  return {create,refresh,setModels:studio.updateModels,open:async(projectId,taskId)=>{if(context?.projectId!==projectId || context?.taskId!==taskId){context={projectId,taskId};epoch++;detailEpoch++;imageEpoch++;rows=[];draft=null;loading=false;createDialog.close();detail.close();confirmDialog.close();}await studio.open(projectId,taskId);await refresh();}};
}
