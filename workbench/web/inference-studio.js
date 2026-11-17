import {request} from './api.js';
const $=s=>document.querySelector(s);
const label=n=>'v'+String(n);
const active=j=>j&&['DISPATCHED','RUNNING'].includes(j.state);
const statusNames={QUEUED:'已保存，等待开始',DISPATCHED:'正在准备推理',RUNNING:'正在推理',SUCCEEDED:'推理完成',FAILED:'推理失败',INTERRUPTED:'推理已中止',CANCELLED:'已取消等待'};

export function createInferenceStudio({onChanged}){
 let ctx=null,epoch=0,page=1,pages=1,total=0,items=[],selected=new Map(),focus=null,job=null,report=null,models=[],busy=false,attempt=null,imageEpoch=0,sourceEpoch=0,jobEpoch=0,polling=false,searchTimer,backends=[];
 const base=()=>`/api/projects/${ctx.projectId}/tasks/${ctx.taskId}`;
 const storeKey=()=>`vision-inference-studio:${ctx.projectId}:${ctx.taskId}`;
 const locked=()=>busy||active(job)||job?.state==='QUEUED'||!!attempt;
 function persist(){try{sessionStorage.setItem(storeKey(),JSON.stringify({attempt,jobId:job?.id||null}));}catch{}}
 function clearStored(){try{sessionStorage.removeItem(storeKey());}catch{}}
 function notice(text,error=false){$('#studio-message').textContent=text;$('#studio-message').className=error?'studio-message error-notice':'studio-message';}
 function conf(){return Number($('#studio-confidence').value);}
 function controls(){
  const freeze=locked();for(const id of ['studio-model','studio-backend','studio-cuda-profile','studio-dataset','studio-upload','studio-nms','studio-source-search','studio-clear'])$('#'+id).disabled=freeze;
  $('#studio-deployment-profile').hidden=$('#studio-backend').value!=='tensorrt';
  const local=$('#studio-dataset').value==='__selected';
  $('#studio-history').disabled=freeze;$('#studio-prev').disabled=freeze||local||page<=1;$('#studio-next').disabled=freeze||local||page>=pages;
  $('#studio-prev').hidden=$('#studio-next').hidden=local||pages<=1;
  for(const n of document.querySelectorAll('#studio-images button,#studio-images input'))n.disabled=freeze;
  const count=[...selected.values()].filter(i=>i.status!=='INVALID').length;
  $('#studio-selection').textContent=`已选 ${selected.size} 张 · ${count} 张可推理`;
  $('#studio-run').disabled=busy||active(job)||(!attempt&&job?.state!=='QUEUED'&&(!count||!$('#studio-model').value||($('#studio-backend').value==='tensorrt'&&!backends.find(b=>b.id==='tensorrt')?.enabled)));
  $('#studio-run').textContent=busy?'正在提交…':active(job)?'推理中…':attempt?'确认并继续本次推理':job?.state==='QUEUED'?'继续本次推理':'开始推理';
  $('#studio-cancel').hidden=!job||!['QUEUED','DISPATCHED','RUNNING'].includes(job.state);$('#studio-cancel').disabled=busy;
  $('#studio-cancel').textContent=active(job)?'中止推理':'取消等待';
  $('#studio-reconnect').hidden=!active(job)||!job.remote?.error;$('#studio-reconnect').disabled=busy;
 }
 function previewUrl(item){return item.kind==='dataset'?`/api/projects/${ctx.projectId}/datasets/${item.dataset_id}/images/${item.id}/view`:`${base()}/inference-inputs/${item.id}/view`;}
 function newDraft(){$('#studio-benchmark').hidden=true;$('#studio-deployment-download').hidden=true;job=null;report=null;attempt=null;jobEpoch++;clearStored();$('#studio-history').value='';$('#studio-confidence').min=$('#studio-confidence-range').min=0.001;$('#studio-result-tag').textContent='原图预览';$('#studio-state').textContent='待推理';$('#studio-downloads').hidden=true;$('#studio-summary').textContent='选择图片后点击“开始推理”，结果会直接显示在这里。';}
 function choose(item){
  if(locked())return;
  if(!selected.has(item.key)&&selected.size>=50){notice('本次最多选择50张图片，请先移除一些图片。',true);return;}
  newDraft();selected.set(item.key,item);focus=item.key;renderImages();draw();controls();notice('已选择图片。调整左下角显示置信度后，点击右下角“开始推理”。');
 }
 function renderImages(){
  const list=$('#studio-images'),scroll=list.scrollTop;list.replaceChildren();const query=$('#studio-source-search').value.trim().toLocaleLowerCase();const visible=$('#studio-dataset').value==='__selected'?[...selected.values()].filter(i=>i.filename.toLocaleLowerCase().includes(query)):items;
  $('#studio-source-page').textContent=$('#studio-dataset').value==='__selected'?`${visible.length} 张本次图片`:`共 ${total} 张 · ${page} / ${pages} 页`;
  for(const item of visible){
   const card=document.createElement('div');card.className='studio-image-card';if(focus===item.key)card.classList.add('current');if(selected.has(item.key))card.classList.add('selected');
   const pick=document.createElement('input');pick.type='checkbox';pick.checked=selected.has(item.key);pick.setAttribute('aria-label',`选择图片：${item.filename}`);
   pick.onchange=()=>{if(locked())return;if(pick.checked){choose(item);return;}newDraft();selected.delete(item.key);if(focus===item.key)focus=selected.keys().next().value||null;renderImages();draw();controls();};
   const button=document.createElement('button');button.type='button';button.className='studio-thumbnail';button.setAttribute('aria-label',`预览并选择：${item.filename}`);if(focus===item.key)button.setAttribute('aria-current','true');
   const thumb=document.createElement('span');thumb.className='studio-image-thumb';
   if(item.status!=='INVALID'){const img=document.createElement('img');img.src=item.kind==='dataset'?`/api/projects/${ctx.projectId}/datasets/${item.dataset_id}/images/${item.id}/thumbnail`:`${base()}/inference-inputs/${item.id}/thumbnail`;img.alt='';img.loading='lazy';thumb.append(img);}else thumb.textContent='坏图';
   button.onclick=()=>{if(report&&selected.has(item.key)){focus=item.key;renderImages();draw();}else choose(item);};
   const text=document.createElement('span');text.className='studio-image-text';const name=document.createElement('span');name.className='studio-image-name';name.textContent=item.filename;name.title=item.filename;
   const state=document.createElement('span');state.className='studio-image-state';const row=report?.images.find(i=>i.id===item.id);state.textContent=item.status==='INVALID'?'图片无效':row?row.status==='SUCCEEDED'?'推理完成':'未完成推理':selected.has(item.key)?'已选':'未选';if(focus===item.key)state.textContent+=' · 当前预览';
   text.append(name,state);button.append(thumb,text);card.append(pick,button);list.append(card);
  }
  list.scrollTop=scroll;
  $('#studio-source-empty').hidden=!!visible.length;$('#studio-source-empty').textContent=$('#studio-dataset').value==='__selected'?'还没有选择图片。可切换数据集，或从本地添加图片。':'当前数据集没有匹配图片，请更换数据集或清除搜索。';controls();
 }
 async function sources(next=1){
  if(!ctx)return;const token=epoch,ticket=++sourceEpoch;page=next;const did=$('#studio-dataset').value;
  if(did==='__selected'){pages=1;renderImages();return;}
  if(!did){items=[];pages=1;renderImages();return;}
  try{const params=new URLSearchParams({page,page_size:48,q:$('#studio-source-search').value.trim(),sort:'name'});
   const data=await request(`/api/projects/${ctx.projectId}/datasets/${did}/images?${params}`);if(token!==epoch||ticket!==sourceEpoch)return;
   pages=data.pages;total=data.total;items=data.images.map(i=>({...i,kind:'dataset',dataset_id:did,key:`dataset:${did}:${i.id}`,status:'READY'}));renderImages();$('#studio-images').scrollTop=0;
  }catch(e){if(token===epoch&&ticket===sourceEpoch)notice(`无法读取图片：${e.message} 可更换数据集或重试搜索。`,true);}
 }
 async function draw(){
  const ticket=++imageEpoch,token=epoch,canvas=$('#studio-canvas'),placeholder=$('#studio-placeholder');canvas.hidden=true;placeholder.hidden=false;
  const item=selected.get(focus);$('#studio-downloads').hidden=true;$('#studio-box-list').replaceChildren();
  if(!item){placeholder.textContent='从左侧选择一张图片，即可预览；点击“开始推理”后，这里显示推理结果。';$('#studio-image-title').textContent='尚未选择图片';return;}
  $('#studio-image-title').textContent=item.filename;
  const row=report?.images.find(i=>i.id===item.id),minimum=report?.snapshot.requested_parameters.conf??0.001,value=conf();
  if(!Number.isFinite(value)||value<minimum||value>1){placeholder.textContent=`显示置信度需在 ${minimum} 至1之间。`;$('#studio-summary').textContent='请修正显示参数后查看当前图片。';return;}
  $('#studio-result-tag').textContent=row?'推理结果':'原图预览';
  if(item.status==='INVALID'||(row&&row.status!=='SUCCEEDED')){placeholder.textContent=`这张图片未能推理：${row?.error||item.error||'尚未处理'}。请检查图片，或在新一轮选择中换一张图片。`;$('#studio-summary').textContent=`${item.filename} · 未完成推理，请检查图片或换一张图片。`;return;}
  const boxes=row?row.predictions.filter(b=>b.confidence>=value):[];
  if(row){
   $('#studio-summary').textContent=`${item.filename} · ${boxes.length?`${boxes.length} 个目标`:'当前置信度下无目标（推理成功）'} · ${row.width} × ${row.height}${report.acceleration&&Number.isFinite(row.processing_ms)?` · 完整处理 ${row.processing_ms.toFixed(2)}ms（含轮廓提取）`:''}`;
   $('#studio-downloads').hidden=false;$('#studio-png').href=`${base()}/inferences/${job.id}/images/${row.id}/overlay?conf=${value}`;$('#studio-json').href=`${base()}/inferences/${job.id}/export?conf=${value}`;
   for(const box of boxes){const li=document.createElement('li');li.textContent=`${box.class_name} · ${(box.confidence*100).toFixed(1)}% · [${box.xyxy.map(v=>v.toFixed(1)).join(', ')}]`;$('#studio-box-list').append(li);}
  }
  const image=new Image();image.src=previewUrl(item);placeholder.textContent='正在读取图片…';
  try{await image.decode();if(ticket!==imageEpoch||token!==epoch)return;
   const scale=Math.min(1,1400/image.naturalWidth);canvas.width=Math.round(image.naturalWidth*scale);canvas.height=Math.round(image.naturalHeight*scale);
   const c=canvas.getContext('2d');c.drawImage(image,0,0,canvas.width,canvas.height);c.lineWidth=2;c.font='14px sans-serif';
   if($('#studio-show-boxes').checked)for(const box of boxes){const [x1,y1,x2,y2]=box.xyxy;c.strokeStyle='#ee852c';c.fillStyle='#ee852c';if(box.segments){c.beginPath();for(const ring of box.segments){ring.points.forEach(([x,y],i)=>i?c.lineTo(x*scale,y*scale):c.moveTo(x*scale,y*scale));c.closePath();}c.fillStyle='#ee852c44';c.fill('evenodd');c.stroke();c.fillStyle='#ee852c';}else c.strokeRect(x1*scale,y1*scale,(x2-x1)*scale,(y2-y1)*scale);c.fillText(`${box.class_name} ${box.confidence.toFixed(2)}`,x1*scale+3,Math.max(16,y1*scale-5));}
   canvas.hidden=false;placeholder.hidden=true;
  }catch{if(ticket===imageEpoch){placeholder.textContent='图片加载失败。请刷新本次记录，或重新选择图片；已有结果不会被删除。';}}
 }
 function updateJobStatus(){
  const state=job?.state;$('#studio-state').textContent=job?statusNames[state]:'待选择';
  if(active(job)){
   const p=job.remote?.progress||{};$('#studio-summary').textContent=`正在处理 ${p.images_completed??0} / ${job.snapshot.sample_count} 张图片。可以等待完成；关闭网页不会重复提交。`;
  }
  if(job?.remote?.error)notice(`${job.remote.error} 当前任务已保留，请重试连接或取消；不会自动创建另一条推理。`,true);
  controls();
 }
 async function load(id,{replace=true}={}){
  if(!ctx||busy)return;const token=epoch,ticket=++jobEpoch;
  try{const data=await request(`${base()}/inferences/${id}`);if(token!==epoch||ticket!==jobEpoch)return;job=data.job;attempt=null;persist();
   if(replace){$('#studio-backend').value=job.snapshot.backend || 'pytorch';$('#studio-cuda-profile').value=job.snapshot.acceleration?.deployment_profile || 'cuda12';selected=new Map(job.snapshot.inputs.map(i=>[`input:${i.id}`,{...i,key:`input:${i.id}`,kind:'input'}]));focus=[...selected.values()].find(i=>i.status==='READY')?.key||selected.keys().next().value;$('#studio-model').value=job.model_id;$('#studio-dataset').value='__selected';$('#studio-source-search').value='';$('#studio-nms').value=job.snapshot.requested_parameters.iou;report=null;}
   updateJobStatus();
   if(job.dispatch?.result?.artifacts?.['report.json']){
    const result=await request(`${base()}/inferences/${id}/report`);if(token!==epoch||ticket!==jobEpoch)return;report=result;
    $('#studio-benchmark').hidden=!result.acceleration;
    if(result.acceleration){const a=result.acceleration;$('#studio-benchmark').textContent=`本机预热后：TensorRT模型推理 ${a.tensorrt.inference_median_ms.toFixed(2)}ms；含常规前后处理 ${a.tensorrt.processing_median_ms.toFixed(2)}ms（P95 ${a.tensorrt.processing_p95_ms.toFixed(2)}ms），原生 ${a.pytorch.processing_median_ms.toFixed(2)}ms。20次对比，未计文件读取、轮廓提取、曝光、取图和通信。完整单图时间见上方；3050/5050速度和FP16效果需现场复核。`;}
    $('#studio-deployment-download').hidden=!job.dispatch?.result?.artifacts?.['deployment.zip'];$('#studio-deployment-download').href=`${base()}/inferences/${id}/artifacts/deployment.zip`;
    const min=job.snapshot.requested_parameters.conf;$('#studio-confidence').min=min;$('#studio-confidence-range').min=min;
    if(conf()<min){$('#studio-confidence').value=min;$('#studio-confidence-range').value=min;}
    const successful=result.images.filter(i=>i.status==='SUCCEEDED').length;
    const resultState=job.state==='INTERRUPTED'?'任务已中断，已保留图片结果':job.state==='FAILED'?'任务失败，已保留图片结果':result.complete?'本批次完成':'已有部分结果';
    notice(`${resultState}：${successful} 张成功、${result.images.length-successful} 张未成功。点击左侧图片查看；调整置信度会立即筛选结果。`);
   }else if(job.state==='QUEUED'){notice('本次图片已保存，尚未使用GPU。点击“继续本次推理”开始，或取消等待。');}
   else if(['FAILED','INTERRUPTED','CANCELLED'].includes(job.state)){notice(job.dispatch?.result?.report?.error||'本次推理没有完整结果。已有输入保留，可点击“新一轮选择”再试。',true);}
   renderImages();await draw();
  }catch(e){if(token===epoch&&ticket===jobEpoch)notice(`无法读取推理记录：${e.message} 请从“已有结果”重新选择。`,true);}
 }
 function updateRows(rows){
  const select=$('#studio-history'),prior=select.value;select.replaceChildren();const none=document.createElement('option');none.value='';none.textContent='查看历史结果';select.append(none);
  for(const j of rows){const o=document.createElement('option');o.value=j.id;o.textContent=`${j.name} · ${statusNames[j.state]}`;select.append(o);}select.value=job?.id||prior;controls();
  const latest=job&&rows.find(j=>j.id===job.id);if(latest&&latest.state!==job.state&&!busy)load(job.id,{replace:false});
 }
 function updateModels(list){
  models=list;const select=$('#studio-model'),prior=select.value;select.replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='选择推理模型';select.append(empty);
  for(const model of models){const option=document.createElement('option');option.value=model.id;option.textContent=`${label(model.number)} · ${model.manifest.training_name}`;option.disabled=!model.available;select.append(option);}
  select.value=models.some(m=>m.id===prior)?prior:models.find(m=>m.available)?.id||'';controls();
 }
 async function chooseModel(id,prior=null){
  if(!ctx||locked())return;newDraft();$('#studio-model').value=id;
  if(prior){selected=new Map(prior.snapshot.inputs.map(i=>[`input:${i.id}`,{...i,kind:'input',key:`input:${i.id}`} ]));focus=selected.keys().next().value;$('#studio-dataset').value='__selected';$('#studio-nms').value=prior.snapshot.requested_parameters.iou;}
  document.dispatchEvent(new CustomEvent('task-workspace-select',{detail:'inference'}));renderImages();draw();notice('模型已选好。选择图片后点击“开始推理”，无需另建配置或打开结果窗口。');
  $('#inference-studio').scrollIntoView({block:'start'});
 }
 async function run(){
  if(busy||active(job)||!ctx)return;
  if(!Number.isFinite(conf())||conf()<0.001||conf()>1){notice('显示置信度需在0.001至1之间，请先修改左下角参数。',true);return;}
  const nms=Number($('#studio-nms').value);
  if(!attempt&&job?.state!=='QUEUED'&&(!Number.isFinite(nms)||nms<0.001||nms>0.95)){$('#studio-nms').closest('details').hidden=false;notice('NMS IoU需在0.001至0.95之间，请展开“推理设置”修改。',true);return;}
  const token=epoch;let failure='';busy=true;report=null;draw();controls();notice('正在保存本次图片和配置…');
  try{
   if(!job||job.state!=='QUEUED'){
    if(!attempt){
     const inputs=[];
     for(const item of selected.values()){
      if(item.kind==='dataset'){
       const value=await request(`${base()}/inference-inputs/from-dataset`,{method:'POST',body:{dataset_id:item.dataset_id,image_id:item.id}});inputs.push(value.input);
      }else inputs.push(item);
      if(token!==epoch)return;
     }
     const ids=[...new Set(inputs.map(i=>i.id))];if(!inputs.some(i=>i.status==='READY'))throw Error('没有可推理的图片，请换一张有效的JPG或PNG。');
     const model=models.find(m=>m.id===$('#studio-model').value);if(!model)throw Error('请选择一个可用模型。');
     attempt={payload:{...($('#studio-backend').value==='tensorrt'?{backend:'tensorrt',deployment_profile:$('#studio-cuda-profile').value}:{}),model_id:model.id,name:`图片推理 ${label(model.number)}`,input_ids:ids,conf:0.001,iou:Number($('#studio-nms').value),idempotency_key:crypto.randomUUID().replaceAll('-','')}};persist();
    }
    const result=await request(`${base()}/inferences`,{method:'POST',body:attempt.payload});if(token!==epoch)return;
    job=result.job;attempt=null;selected=new Map(job.snapshot.inputs.map(i=>[`input:${i.id}`,{...i,key:`input:${i.id}`,kind:'input'}]));focus=selected.keys().next().value;$('#studio-dataset').value='__selected';persist();
   }
   notice('正在连接计算目标并开始推理，请等待；不会训练模型。');
   const result=await request(`${base()}/inferences/${job.id}/remote-run`,{method:'POST',body:{}});if(token!==epoch)return;job=result.job;persist();updateJobStatus();
  }catch(e){if(token===epoch){if(!job&&['MODEL_DAMAGED','INFERENCE_INPUT_DAMAGED','NO_VALID_INFERENCE_INPUT','INVALID_INFERENCE_THRESHOLD','INFERENCE_BATCH_SIZE','INVALID_MODEL_ID'].includes(e.code)){attempt=null;persist();}failure=`${e.message} ${job?'当前任务已保留，点击“继续本次推理”重试连接；不会新建重复任务。':attempt?'保存状态未确认，请再次确认同一请求；不会使用新的提交标识。':'尚未开始GPU推理，请检查左侧图片和参数后重试。'}`;notice(failure,true);}}
  finally{if(token===epoch){busy=false;renderImages();updateJobStatus();await onChanged();if(job)await load(job.id,{replace:false});if(failure)notice(failure,true);}}
 }
 $('#studio-run').onclick=run;$('#studio-reconnect').onclick=run;
 $('#studio-cancel').onclick=async()=>{
  if(!job||busy)return;if(active(job)&&!window.confirm('中止当前推理并保留已完成结果？确认进程与设备释放后才会结束。'))return;
  const id=job.id,token=epoch;busy=true;controls();try{await request(`${base()}/inferences/${id}/${active(job)?'remote-stop':'cancel'}`,{method:'POST',body:{}});}catch(e){if(token===epoch)notice(e.message,true);}finally{if(token===epoch){busy=false;await onChanged();await load(id);}}
 };
 $('#studio-clear').onclick=()=>{if(locked())return;newDraft();selected.clear();focus=null;$('#studio-history').value='';if($('#studio-dataset').value==='__selected')$('#studio-dataset').value=[...$('#studio-dataset').options].find(o=>o.value!=='__selected')?.value||'__selected';sources();draw();controls();notice('已开始新一轮选择。历史结果仍可从右上方查看。');};
 $('#studio-backend').onchange=$('#studio-cuda-profile').onchange=()=>{if(locked())return;newDraft();draw();controls();notice($('#studio-backend').value==='tensorrt'?'TensorRT首次会优化模型并缓存，完成后可下载产线包。FP16效果请使用代表性样本复核。':'已选择原生推理。');};
 document.addEventListener('inference-accelerate-model',event=>{if(!ctx || !models.some(m=>m.id===event.detail))return;if(locked()){notice('请先结束或取消当前推理，再准备TensorRT模型。',true);return;}newDraft();$('#studio-model').value=event.detail;$('#studio-backend').value='tensorrt';controls();notice('已选择TensorRT加速，请选择验证图片和产线配置，再点击开始推理。');});
 $('#studio-model').onchange=()=>{if(locked())return;newDraft();draw();notice('模型已改变，右侧为原图；点击“开始推理”生成新结果。');controls();};
 $('#studio-nms').onchange=()=>{if(locked())return;newDraft();draw();notice('NMS参数已改变，需要重新推理；当前只显示原图。');controls();};
 $('#studio-dataset').onchange=()=>{if(locked())return;if(job||report){newDraft();selected.clear();focus=null;draw();}sources();};
 $('#studio-prev').onclick=()=>sources(page-1);$('#studio-next').onclick=()=>sources(page+1);
 function search(){clearTimeout(searchTimer);if(!locked())sources(1);}
 $('#studio-source-search').oninput=e=>{clearTimeout(searchTimer);sourceEpoch++;if(!e.isComposing)searchTimer=setTimeout(search,250);};
 $('#studio-source-search').oncompositionend=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(search,250);};
 $('#studio-source-search').onkeydown=e=>{if(e.key==='Enter'&&!e.isComposing){e.preventDefault();search();}};
 $('#studio-history').onchange=()=>{if(!locked()&&$('#studio-history').value)load($('#studio-history').value);};
 for(const id of ['studio-confidence','studio-confidence-range'])$('#'+id).oninput=()=>{
  const value=$('#'+id).value;$('#studio-confidence').value=value;$('#studio-confidence-range').value=value;draw();
 };
 $('#studio-show-boxes').onchange=draw;
 $('#studio-upload').onchange=async e=>{
  if(locked()||!ctx)return;const files=[...e.target.files];if(selected.size+files.length>50){notice('每次最多50张图片，请减少选择。',true);return;}
  const token=epoch;newDraft();busy=true;controls();let failures=0;
  try{for(let i=0;i<files.length;i++){
   notice(`正在添加本地图片 ${i+1}/${files.length}，不会使用GPU。`);const f=files[i];
   try{const r=await fetch(`${base()}/inference-inputs`,{method:'POST',headers:{'Content-Type':'application/octet-stream','X-File-Name':encodeURIComponent(f.name)},body:f});const data=await r.json();if(!r.ok)throw Error(data.error?.message||'图片未保存');if(token!==epoch)return;const item={...data.input,kind:'input',key:`input:${data.input.id}`};selected.set(item.key,item);focus??=item.key;}catch(err){failures++;notice(`${f.name}：${err.message}`,true);}
  }}finally{if(token===epoch){busy=false;$('#studio-dataset').value='__selected';renderImages();draw();notice(failures?`${failures} 张图片未能添加，其余图片已保留。请检查图片或重新选择。`:'图片已添加，点击“开始推理”即可查看结果。',!!failures);}}
 };
 setInterval(async()=>{if(ctx&&!busy&&!polling&&active(job)&&$('#inference-studio').getClientRects().length){polling=true;try{await load(job.id,{replace:false});await onChanged();}finally{polling=false;}}},3000);
 return {chooseModel,load,updateRows,updateModels,async open(projectId,taskId){
  if(ctx?.projectId===projectId&&ctx?.taskId===taskId)return;
  clearTimeout(searchTimer);ctx={projectId,taskId};const token=++epoch;sourceEpoch++;jobEpoch++;imageEpoch++;job=null;report=null;attempt=null;busy=false;selected.clear();focus=null;items=[];total=0;page=pages=1;
  $('#studio-images').replaceChildren();$('#studio-confidence').value=$('#studio-confidence-range').value=0.25;$('#studio-confidence').min=$('#studio-confidence-range').min=0.001;$('#studio-show-boxes').checked=true;
  try{const [modelData,datasets,task]=await Promise.all([request(`${base()}/models`),request(`/api/projects/${projectId}/datasets`),request(base())]);if(token!==epoch)return;
   $('#studio-show-boxes').parentElement.lastChild.textContent=task.task.task_type==='segment'?'显示分割轮廓':'显示预测框';models=modelData.models;const select=$('#studio-model');select.replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='选择推理模型';select.append(empty);
   for(const model of models){const option=document.createElement('option');option.value=model.id;option.textContent=`${label(model.number)} · ${model.manifest.training_name}`;option.disabled=!model.available;select.append(option);}select.value=models.find(m=>m.available)?.id||'';
   const source=$('#studio-dataset');source.replaceChildren();const chosen=document.createElement('option');chosen.value='__selected';chosen.textContent='本次图片 / 本地图片';source.append(chosen);for(const d of datasets.datasets){const o=document.createElement('option');o.value=d.id;o.textContent=`${d.name} · ${d.image_count} 张`;source.append(o);}source.value=datasets.datasets.some(d=>d.id===task.task.dataset_id)?task.task.dataset_id:'__selected';
   $('#studio-context').textContent=`${$('#detail-name').textContent} / ${task.task.name} · ${task.task.task_type==='segment'?'实例分割':'目标检测'}`;
   const options=select.value?await request(`${base()}/inferences/options?model_id=${select.value}`):null;if(token!==epoch)return;backends=options?.backends || [];$('#studio-backend option[value="tensorrt"]').disabled=!backends.find(b=>b.id==='tensorrt')?.enabled;$('#studio-run-note').textContent=options?`使用 ${options.target.name} 处理所选图片。`:'请先完成一次训练，再选择生成的模型。';
   await sources();draw();notice('从左侧选择图片，右侧可先预览；点击“开始推理”才会使用GPU。');
   let stored;try{stored=JSON.parse(sessionStorage.getItem(storeKey())||'null');}catch{}
   if(stored?.jobId)await load(stored.jobId);else if(stored?.attempt?.payload){attempt=stored.attempt;$('#studio-model').value=attempt.payload.model_id;$('#studio-backend').value=attempt.payload.backend || 'pytorch';$('#studio-cuda-profile').value=attempt.payload.deployment_profile || 'cuda12';notice('上次保存状态尚未确认。点击右下角继续同一次请求，不会重复创建任务。',true);}
  }catch(e){if(token===epoch)notice(`工作区加载失败：${e.message} 请刷新页面，已保存记录不会丢失。`,true);}finally{if(token===epoch)controls();}
 }};
}
