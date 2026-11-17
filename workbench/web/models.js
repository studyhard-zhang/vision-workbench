import { request } from './api.js';
import { createEvaluationWorkspace } from './evaluations.js';
import { createInferenceWorkspace } from './inferences.js';
const $=s=>document.querySelector(s);
const label=n=>'v'+String(n);
const dataLabel=n=>'V'+String(n).padStart(3,'0');
const time=v=>new Date(v).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});

export function createModelWorkspace(){
  const evaluations=createEvaluationWorkspace({onChanged:()=>refresh()});
  const inferences=createInferenceWorkspace();
  let context=null,epoch=0;
  const dialog=$('#model-detail-dialog');
  const base=c=>`/api/projects/${c.projectId}/tasks/${c.taskId}/models`;
  function error(message){$('#model-error').textContent=message;$('#model-error').hidden=!message;}
  async function view(id){
    const token=epoch,target=context;
    try{
      const {model}=await request(`${base(target)}/${id}`);if(token!==epoch)return;
      const m=model.manifest,report=m.receipt.report,segment=m.training_snapshot.task_type==='segment';
      $('#model-detail-title').textContent=`模型 ${label(model.number)}`;
      $('#model-detail-status').textContent=model.available?'文件校验通过 · 可直接进行图片推理':`文件异常，已阻止下载：${model.integrity.issues.join('、')}`;
      if(m.training_snapshot.initial_model)$('#model-detail-status').textContent+=` · 基于模型 v${m.training_snapshot.initial_model.number} 训练`;
      $('#model-detail-summary').textContent=`来源训练：${m.training_name}\n训练运行 ID：${model.training_run_id}\n数据版本：${dataLabel(m.data_version_number)} · ${m.task_data_version_id}\n生成时间：${time(model.created_at)}\n模型 ID：${model.id}\n类别：${m.classes.map(c=>`${c.index}: ${c.name}`).join('、')}\n实际训练：${report.epochs_completed} 轮 / ${report.batches_completed} 批次 · 优化器 ${report.actual_optimizer}\n默认权重：${m.default_artifact}\n${m.load_check.description}`;
      const scores=report.metrics||{},suffix=segment?'M':'B';for(const [id,key] of [['precision','precision'],['recall','recall'],['map','mAP50']]){const value=scores[`metrics/${key}(${suffix})`];$('#model-key-'+id).textContent=typeof value==='number'?`${(value*100).toFixed(2)}%`:'未记录';}const strict=scores[`metrics/mAP50-95(${suffix})`];$('#model-key-map50').textContent=`mAP@0.5–0.95：${typeof strict==='number'?(strict*100).toFixed(2)+'%':'未记录'}`;$('#model-key-context').textContent=`${segment?'掩膜(M)':'检测框(B)'} · best.pt回传内置验证；不是末轮指标或独立测试结果。`;
      $('#model-selection-rule').textContent=m.selection_rule.description;
      $('#model-evaluation-status').textContent=model.evaluation_status==='EVALUATED'?'已有独立评估记录，请在下方评估列表选择报告；验证集结果不等同测试集表现。':m.evaluation.reason;
      $('#model-start-evaluation').onclick=()=>{dialog.close();evaluations.create(id);};$('#model-start-evaluation').disabled=!model.available;$('#model-start-evaluation').hidden=true;
      $('#model-start-inference').onclick=()=>{dialog.close();document.dispatchEvent(new CustomEvent('task-workspace-select',{detail:'inference'}));inferences.create(id);};$('#model-start-inference').disabled=!model.available;$('#model-start-inference').hidden=false;
      let iterate=$('#model-train-next');if(!iterate){iterate=document.createElement('button');iterate.id='model-train-next';iterate.type='button';iterate.className='button primary';$('#model-start-inference').before(iterate);}iterate.textContent='基于此模型训练下一版';iterate.disabled=!model.available;iterate.onclick=()=>{dialog.close();document.dispatchEvent(new CustomEvent('task-workspace-select',{detail:'training'}));document.dispatchEvent(new CustomEvent('training-from-model',{detail:id}));};
      $('#model-detail-manifest').textContent=JSON.stringify(m,null,2);
      $('#model-manifest-hash').textContent=`来源清单 SHA-256：${model.manifest_sha256}`;
      const body=$('#model-artifacts');body.replaceChildren();
      for(const [name,spec] of Object.entries(m.artifacts)){
        const row=document.createElement('tr');
        for(const value of [name,`${spec.bytes.toLocaleString()} 字节`,spec.sha256]){const cell=document.createElement('td');cell.textContent=value;if(value===spec.sha256)cell.className='training-hash';row.append(cell);}
        const cell=document.createElement('td');if(model.available){const link=document.createElement('a');link.className='button secondary';link.href=`${base(target)}/${id}/artifacts/${name}`;link.download='';link.textContent=`下载 ${name}`;cell.append(link);}else cell.textContent='文件异常';row.append(cell);body.append(row);
      }
      const link=$('#model-manifest-download');link.hidden=!model.available;link.href=`${base(target)}/${id}/artifacts/manifest.json`;
      $('#model-artifact-details').open=false;
      const delivery=$('#model-delivery-download');delivery.hidden=!model.available;delivery.href=`${base(target)}/${id}/${segment?'artifacts/best.pt':'delivery'}`;delivery.textContent=segment?'下载分割模型 best.pt':'下载 Python 模型包';
      let accelerate=$('#model-tensorrt');if(!accelerate){accelerate=document.createElement('button');accelerate.id='model-tensorrt';accelerate.type='button';accelerate.className='button primary';$('#model-start-inference').after(accelerate);}accelerate.textContent='TensorRT 加速与产线导出';accelerate.disabled=!model.available;accelerate.onclick=()=>{dialog.close();document.dispatchEvent(new CustomEvent('task-workspace-select',{detail:'inference'}));document.dispatchEvent(new CustomEvent('inference-accelerate-model',{detail:id}));};
      dialog.showModal();
    }catch(e){if(token===epoch)error(e.message);}
  }
  async function refresh(){
    if(!context)return;const token=epoch,target=context;error('');
    try{
      const {models}=await request(base(target));if(token!==epoch)return;
      inferences.setModels(models);
      const list=$('#model-list');list.replaceChildren();$('#model-empty').hidden=Boolean(models.length);
      for(const model of models){
        const segment=model.manifest.training_snapshot.task_type==='segment';const card=document.createElement('article');card.className='training-row';
        const body=document.createElement('div'),title=document.createElement('h4'),info=document.createElement('p'),status=document.createElement('p');
        title.textContent=`${label(model.number)} · ${model.manifest.training_name}`;
        info.textContent=`数据 ${dataLabel(model.manifest.data_version_number)} · ${model.manifest.classes.length} 类 · ${model.manifest.receipt.report.epochs_completed} 轮 · 运行 ${model.training_run_id.slice(0,8)} · ${time(model.created_at)}`;
        status.textContent=model.available?'文件校验通过 · 可直接进行图片推理':`文件异常，禁止下载：${model.integrity.issues.join('、')}`;
        body.append(title,info,status);const button=document.createElement('button');button.className='button secondary';button.textContent='查看模型版本';button.onclick=()=>view(model.id);const evaluate=document.createElement('button');evaluate.className='button secondary';evaluate.textContent='独立评估此版本';evaluate.disabled=!model.available;evaluate.hidden=false;evaluate.onclick=()=>evaluations.create(model.id);const actions=document.createElement('div');actions.className='training-row-actions';const infer=document.createElement('button');infer.className='button secondary';infer.textContent='图片推理此版本';infer.disabled=!model.available;infer.hidden=false;infer.onclick=()=>{document.dispatchEvent(new CustomEvent('task-workspace-select',{detail:'inference'}));inferences.create(model.id);};actions.append(button,evaluate,infer);card.append(body,actions);list.append(card);
        if(model.available){const delivery=document.createElement('a');delivery.className='button primary';delivery.href=`${base(target)}/${model.id}/${segment?'artifacts/best.pt':'delivery'}`;delivery.download='';delivery.textContent=segment?'下载分割模型 best.pt':'下载 Python 模型包';delivery.setAttribute('aria-label',`下载 ${label(model.number)} ${segment?'分割模型 best.pt':'Python 模型包'}`);actions.prepend(delivery);}
      }
    }catch(e){if(token===epoch)error(e.message);}
  }
  $('#model-refresh').onclick=async()=>{try{const r=await request(`${base(context)}/sync`,{method:'POST',body:{}});await refresh();if(r.errors.length)error('部分模型文件尚未准备完成：'+r.errors.map(e=>e.message).join('；'));}catch(e){error(e.message);}};$('#model-detail-close').onclick=()=>dialog.close();
  return {view,refresh,open:async(projectId,taskId)=>{context={projectId,taskId};epoch++;dialog.close();$('#model-list').replaceChildren();const {task}=await request(`/api/projects/${projectId}/tasks/${taskId}`);document.querySelector('section[aria-label="独立模型评估"]').hidden=true;await inferences.open(projectId,taskId);await refresh();}};
}
