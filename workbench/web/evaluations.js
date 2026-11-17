import { request } from './api.js';
const $=s=>document.querySelector(s);
const labels={QUEUED:'待执行',DISPATCHED:'已派发 · 目标占用中',RUNNING:'评估运行中',SUCCEEDED:'评估完成',FAILED:'评估失败',INTERRUPTED:'已中断',CANCELLED:'已取消'};
const number=v=>v===null || v===undefined?'不可用':Number(v).toFixed(4);
const modelName=n=>'v'+String(n);
const dataName=n=>'V'+String(n).padStart(3,'0');
function button(text,action,disabled=false){const b=document.createElement('button');b.className='button secondary';b.textContent=text;b.disabled=disabled;b.onclick=action;return b;}

export function createEvaluationWorkspace({onChanged=()=>{}}={}){
  let context=null,epoch=0,rows=[],draft=null,busy=false,loading=false,confirmation=null,viewed=null,report=null,imageEpoch=0;
  const form=$('#evaluation-create-form'),dialog=$('#evaluation-create-dialog'),detail=$('#evaluation-detail-dialog'),confirmDialog=$('#evaluation-confirm-dialog');
  const base=c=>`/api/projects/${c.projectId}/tasks/${c.taskId}/evaluations`;
  function error(message){$('#evaluation-error').textContent=message;$('#evaluation-error').hidden=!message;}
  function controls(){
    $('#evaluation-refresh').disabled=busy || loading;
    $('#evaluation-create-submit').disabled=busy || !draft;
    $('#evaluation-create-close').disabled=busy;
    $('#evaluation-confirm-submit').disabled=busy;
    $('#evaluation-confirm-close').disabled=busy;
  }
  function render(){
    const list=$('#evaluation-list');list.replaceChildren();$('#evaluation-empty').hidden=!!rows.length;
    for(const job of rows){
      const row=document.createElement('article');row.className='training-row';const body=document.createElement('div'),title=document.createElement('h4'),info=document.createElement('p'),actions=document.createElement('div');actions.className='training-row-actions';
      const s=job.snapshot;title.textContent=job.name;info.textContent=`${modelName(s.model_number)} · ${dataName(s.data_version_number)} · ${s.subset==='val'?'验证集':'测试集'} ${s.sample_count} 张 · ${labels[job.state]}${job.remote?' · SSH '+job.remote.phase:''}`;
      body.append(title,info);actions.append(button('查看评估详情',()=>view(job.id),busy));
      if(job.state==='QUEUED')actions.append(button('SSH运行评估',()=>confirmRun(job,'remote-run'),busy),button('取消等待',()=>operate(job,'cancel'),busy));
      if(['DISPATCHED','RUNNING'].includes(job.state))actions.append(button('恢复评估连接',()=>confirmRun(job,'remote-run'),busy),button('中止评估',()=>confirmRun(job,'remote-stop'),busy));
      if(['FAILED','INTERRUPTED','CANCELLED','SUCCEEDED'].includes(job.state))actions.append(button('按配置新建评估',()=>create(job.model_id,job),busy));
      row.append(body,actions);list.append(row);
    }
    controls();
  }
  async function refresh(){
    if(!context || loading)return;const token=epoch,target=context;loading=true;controls();
    try{const result=await request(base(target));if(token!==epoch)return;rows=result.jobs;render();}
    catch(e){if(token===epoch)error(e.message);}
    finally{if(token===epoch){loading=false;controls();}}
  }
  async function create(modelId,prior=null){
    if(!context || busy)return;const token=epoch,target=context;error('');
    try{
      const options=await request(`${base(target)}/options?model_id=${modelId}`);if(token!==epoch)return;
      if(!options.model.available)throw new Error('模型文件校验不通过，不能创建评估。');
      const s=prior?.snapshot;draft={modelId,key:crypto.randomUUID().replaceAll('-',''),target,token};
      $('#evaluation-create-title').textContent=`独立评估 ${modelName(options.model.number)}`;
      $('#evaluation-create-source').textContent=`权重：best.pt · ${options.model.manifest.artifacts['best.pt'].sha256}\n冻结数据：${dataName(options.model.manifest.data_version_number)} · ${options.model.task_data_version_id}\n计算目标：${options.target.name}\n${options.notice}`;
      const select=$('#evaluation-subset');select.replaceChildren();
      for(const subset of options.subsets){const option=document.createElement('option');option.value=subset.id;option.textContent=`${subset.id==='val'?'验证集':'测试集'} · ${subset.count} 张${subset.count?'':'（不可用）'}`;option.disabled=!subset.count;select.append(option);}
      select.value=s?.subset || options.subsets.find(s=>s.count)?.id || '';
      $('#evaluation-name').value=`评估 ${modelName(options.model.number)}`;
      $('#evaluation-size-note').textContent=`沿用模型训练输入尺寸 ${options.defaults.imgsz} / batch1 / FP32，不增强。`;$('#evaluation-conf').value=s?.requested_parameters.conf ?? options.defaults.conf;
      $('#evaluation-iou').value=s?.requested_parameters.iou ?? options.defaults.iou;
      $('#evaluation-create-error').hidden=true;controls();dialog.showModal();
    }catch(e){if(token===epoch)error(e.message);}
  }
  form.onsubmit=async event=>{
    event.preventDefault();if(!draft || busy || !form.reportValidity())return;const current=draft;busy=true;controls();
    try{
      await request(base(current.target),{method:'POST',body:{model_id:current.modelId,idempotency_key:current.key,name:$('#evaluation-name').value.trim(),subset:$('#evaluation-subset').value,conf:Number($('#evaluation-conf').value),iou:Number($('#evaluation-iou').value)}});
      if(current.token!==epoch)return;draft=null;dialog.close();await refresh();await onChanged();
    }catch(e){if(current.token===epoch){$('#evaluation-create-error').textContent=e.message;$('#evaluation-create-error').hidden=false;}}
    finally{busy=false;controls();render();}
  };
  function confirmRun(job,action){
    if(busy)return;confirmation={job,action,target:context,token:epoch};const s=job.snapshot;
    $('#evaluation-confirm-error').textContent='';
    $('#evaluation-confirm-title').textContent=action==='remote-stop'?'中止原评估进程':'运行或恢复独立评估';
    $('#evaluation-confirm-message').textContent=`${job.name}\n${modelName(s.model_number)} / best.pt · ${s.weights_sha256}\n${dataName(s.data_version_number)} · ${s.subset} ${s.sample_count} 张\n${s.target.name} · ${s.requested_parameters.imgsz} / batch1 / FP32\n最低置信度 ${s.requested_parameters.conf}，NMS IoU ${s.requested_parameters.iou}\n${action==='remote-stop'?'核对并停止原进程，收回中断结果；释放设备锁后才放行下一任务。':'首次运行使用GPU执行一次评估，不训练。恢复连接只核对原进程与结果，不重新评估。'}`;
    controls();confirmDialog.showModal();
  }
  async function operate(job,action){
    if(busy)return;const token=epoch,target=context;busy=true;controls();error('');
    try{await request(`${base(target)}/${job.id}/${action}`,{method:'POST',body:{}});if(token!==epoch)return;confirmDialog.close();await refresh();await onChanged();}
    catch(e){if(token===epoch){error(e.message);$('#evaluation-confirm-error').textContent=e.message;}}
    finally{busy=false;controls();if(token===epoch)render();}
  }
  $('#evaluation-confirm-submit').onclick=()=>{if(confirmation?.token===epoch)operate(confirmation.job,confirmation.action);};
  $('#evaluation-confirm-close').onclick=()=>confirmDialog.close();$('#evaluation-create-close').onclick=()=>dialog.close();
  confirmDialog.addEventListener('cancel',e=>{if(busy)e.preventDefault();});dialog.addEventListener('cancel',e=>{if(busy)e.preventDefault();});
  async function drawImage(){
    const token=++imageEpoch,ctx=context,data=report;if(!data)return;
    const entry=data.images.find(i=>i.id===$('#evaluation-image').value);if(!entry)return;
    const image=new Image();image.src=`/api/projects/${ctx.projectId}/tasks/${ctx.taskId}/versions/${data.snapshot.data_version_id}/images/${entry.id}`;
    try{await image.decode();if(token!==imageEpoch || report!==data)return;
      const canvas=$('#evaluation-canvas'),scale=Math.min(1,1000/entry.width);canvas.width=Math.round(entry.width*scale);canvas.height=Math.round(entry.height*scale);const c=canvas.getContext('2d');c.drawImage(image,0,0,canvas.width,canvas.height);c.lineWidth=2;c.font='13px sans-serif';
      const paint=(box,text,color,segments=null)=>{c.strokeStyle=color;c.fillStyle=color;if(segments){c.beginPath();for(const ring of segments){ring.points.forEach(([x,y],i)=>i?c.lineTo(x*scale,y*scale):c.moveTo(x*scale,y*scale));c.closePath();}c.fillStyle=color+'44';c.fill('evenodd');c.stroke();c.fillStyle=color;}else c.strokeRect(box[0]*scale,box[1]*scale,(box[2]-box[0])*scale,(box[3]-box[1])*scale);c.fillText(text,box[0]*scale+2,Math.max(14,box[1]*scale-3));};
      if($('#evaluation-show-truth').checked)for(const b of entry.truth)paint([b.x,b.y,b.x+b.width,b.y+b.height],'GT '+data.snapshot.classes.find(c=>c.id===b.class_id)?.name,'#16863d',b.points?[{points:b.points,hole:false}]:null);
      const visible=entry.predictions.filter(b=>b.confidence>=Number($('#evaluation-display-conf').value));
      if($('#evaluation-show-prediction').checked)for(const b of visible)paint(b.xyxy,`${data.snapshot.classes[b.class_index]?.name} ${b.confidence.toFixed(3)}`,'#e76c17',b.segments);
      $('#evaluation-image-description').textContent=`${entry.filename} · ${entry.width} × ${entry.height} · 真值 ${entry.truth.length} 个目标 · 当前显示预测 ${visible.length} 个目标。绿色真值，橙色预测；坐标对应冻结原图。`;
    }catch{if(token===imageEpoch)$('#evaluation-image-description').textContent='冻结图片加载失败，请检查原数据完整性。';}
  }
  async function view(id){
    if(!context)return;const token=epoch,target=context;error('');
    try{
      const {job}=await request(`${base(target)}/${id}`);if(token!==epoch)return;viewed=id;report=null;imageEpoch++;
      const s=job.snapshot;$('#evaluation-detail-title').textContent=job.name;$('#evaluation-detail-status').textContent=`${labels[job.state]}${job.remote?' · SSH '+job.remote.phase:''}`;
      $('#evaluation-detail-source').textContent=`模型 ${modelName(s.model_number)} · ${job.model_id}\n数据 ${dataName(s.data_version_number)} · ${s.data_version_id}\n${s.subset==='val'?'冻结验证集':'冻结测试集'} ${s.sample_count} 张 · best.pt SHA ${s.weights_sha256}\n${s.requested_parameters.imgsz} / batch1 / FP32 · 最低置信度 ${s.requested_parameters.conf} · NMS IoU ${s.requested_parameters.iou}`;
      const failure=job.dispatch?.result?.report,diagnostic=failure?.diagnostic;
      $('#evaluation-detail-error').textContent=[job.remote?.error || failure?.error || '',diagnostic?.title,diagnostic?.next_step].filter(Boolean).join('\n');
      $('#evaluation-progress').textContent=JSON.stringify(job.remote?.progress || {},null,2);
      $('#evaluation-snapshot').textContent=JSON.stringify(s,null,2);
      $('#evaluation-report-panel').hidden=true;$('#evaluation-downloads').replaceChildren();
      const artifacts=job.dispatch?.result?.artifacts || {};
      for(const name of Object.keys(artifacts)){const a=document.createElement('a');a.className='button secondary';a.textContent=`下载 ${name}`;a.download='';a.href=`${base(target)}/${id}/artifacts/${name}`;$('#evaluation-downloads').append(a);}
      if(job.state==='SUCCEEDED'){
        const data=await request(`${base(target)}/${id}/report`);if(token!==epoch || viewed!==id)return;report=data;
        $('#evaluation-report-panel').hidden=false;$('#evaluation-metrics').textContent=`${data.metric_scope==='mask'?'分割掩膜 · ':''}Precision ${number(data.metrics.precision)} · Recall ${number(data.metrics.recall)} · mAP@0.5 ${number(data.metrics.map50)} · mAP@0.5:0.95 ${number(data.metrics.map50_95)}`;
        const semanticLabels={matrix_scope:'混淆矩阵口径',ap:'AP口径',confusion_confidence:'混淆矩阵置信度',confusion_match_iou:'混淆矩阵匹配IoU',matrix_axes:'矩阵坐标轴',precision_recall:'P/R口径',subset:'数据范围'};
        $('#evaluation-semantics').textContent=Object.entries(data.metric_semantics).map(([k,v])=>`${semanticLabels[k] || k}：${v}`).join('\n');
        const body=$('#evaluation-class-table');body.replaceChildren();for(const c of data.per_class){const tr=document.createElement('tr');for(const v of [c.name,c.instances,...['precision','recall','map50','map50_95'].map(k=>number(c.metrics[k]))]){const td=document.createElement('td');td.textContent=v;tr.append(td);}body.append(tr);}
        $('#evaluation-timing').textContent=JSON.stringify(data.timing,null,2);$('#evaluation-missing').textContent=data.missing.join('\n');
        const plots=$('#evaluation-plots');plots.replaceChildren();for(const name of Object.keys(artifacts).filter(n=>n.endsWith('.png'))){const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.src=`${base(target)}/${id}/artifacts/${name}`;img.alt=name;img.style.width='100%';caption.textContent=name;figure.append(img,caption);plots.append(figure);}
        $('#evaluation-matrix').textContent=data.confusion_matrix?JSON.stringify(data.confusion_matrix):'运行库未提供混淆矩阵。';
        const select=$('#evaluation-image');select.replaceChildren();for(const image of data.images){const o=document.createElement('option');o.value=image.id;o.textContent=image.filename;select.append(o);}
        $('#evaluation-display-conf').value=Math.max(0.25,s.requested_parameters.conf);await drawImage();
      }
      if(token===epoch && !detail.open)detail.showModal();
    }catch(e){if(token===epoch)error(e.message);}
  }
  $('#evaluation-image').onchange=drawImage;$('#evaluation-display-conf').oninput=drawImage;$('#evaluation-show-truth').onchange=drawImage;$('#evaluation-show-prediction').onchange=drawImage;
  $('#evaluation-detail-close').onclick=()=>{detail.close();imageEpoch++;};$('#evaluation-detail-refresh').onclick=()=>view(viewed);$('#evaluation-refresh').onclick=refresh;
  setInterval(()=>{if(context && !busy && !loading && $('#evaluation-list').getClientRects().length && rows.some(j=>['DISPATCHED','RUNNING'].includes(j.state)))refresh();},4000);
  return {create,view,refresh,open:async(projectId,taskId)=>{if(context?.projectId!==projectId || context?.taskId!==taskId){context={projectId,taskId};epoch++;rows=[];draft=null;loading=false;imageEpoch++;dialog.close();detail.close();confirmDialog.close();}await refresh();}};
}
