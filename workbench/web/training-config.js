import { request } from './api.js';
const $=s=>document.querySelector(s);
const ns='http://www.w3.org/2000/svg';
function node(name,attrs={}){const e=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;}

export function createTrainingConfiguration({onChange}){
  let context=null,options=null,manifest=null,token=0,sample=0,recommendation=null,recommendationToken=0,pendingRecommendation=false,modelToken=0,modelPending=false;
  const ids=['fliplr','flipud','seed','patience','optimizer','lr0','momentum'];
  function value(){

    const profile=$('#training-profile').value;if(profile==='legacy')return null;if(profile==='official26')return {profile};if(profile==='custom26'){const result={profile};const size=Number($('#training-imgsz').value),batch=Number($('#training-batch').value);if(size!==640)result.imgsz=size;if(batch!==16)result.batch=batch;return result;}
    const result={profile};if(profile==='precision'){result.imgsz=Number($('#training-imgsz').value);result.batch=Number($('#training-batch').value);}if(options?.task_type==='segment')return {...result,fliplr:0,flipud:0,seed:42,patience:100,optimizer:'auto'};for(const id of ids){if(['lr0','momentum'].includes(id) && $('#training-optimizer').value==='auto')continue;result[id]=id==='optimizer'?$('#training-'+id).value:Number($('#training-'+id).value);}return result;
  }
  function verified(){if(modelPending || (options?.training_mode || 'standard')!==$('#training-model-mode').value || !options?.training_modes?.find(m=>m.id===$('#training-model-mode').value)?.enabled)return false;if($('#training-parameter-mode').value==='recommended' && (pendingRecommendation || !recommendation))return false;const cfg=value();const profile=options?.configuration_profiles.find(p=>p.id===cfg?.profile);return cfg?Boolean(profile?.enabled && profile.verified_optimizers.includes(cfg.optimizer || 'auto')):options?.legacy_enabled!==false;}
  function invalidate(){token++;$('#augmentation-result').hidden=true;$('#augmentation-error').hidden=true;$('#augmentation-preview').disabled=false;$('#augmentation-next').disabled=false;$('#augmentation-status').textContent='配置或图片变更后请重新预览。';}
  function update(){
    invalidate();const cfg=value();$('#training-config-fields').hidden=!cfg;
    const high=$('#training-model-mode').value==='high_accuracy';$('#training-mode-help').textContent=high?'高精度模式使用更大容量模型，训练时间更长、显存需求更高；实际效果以验证结果为准。':'通用模式兼顾训练效率与模型表现。';$('#training-weight-info').textContent=`预训练模型：${options?.task_type==='segment'?'SegDL':'DetectDL'} · ${high?'高精度模式':'通用模式'}`;
    const segment=options?.task_type==='segment';$('#training-profile').disabled=false;$('#training-profile option[value="legacy"]').hidden=segment;$('#training-profile option[value="geometry512"]').hidden=segment;
    if(segment || ['official26','custom26'].includes(cfg?.profile))$('#training-config-fields').hidden=true;
    $('#training-legacy-help').hidden=Boolean(cfg);
    for(const p of options?.configuration_profiles || []){const option=$('#training-profile').querySelector(`option[value="${p.id}"]`);if(option)option.hidden=!p.enabled && cfg?.profile!==p.id;}
    const profile=options?.configuration_profiles.find(p=>p.id===cfg?.profile);
    const size=cfg?.imgsz||profile?.imgsz||640,batch=cfg?.batch??profile?.batch??1;$('#training-imgsz').disabled=$('#training-batch').disabled=!['precision','custom26'].includes(cfg?.profile);if(!['precision','custom26'].includes(cfg?.profile)){$('#training-imgsz').value=size;$('#training-batch').value=batch;}
    $('#training-fixed-values').replaceChildren(...[`输入尺寸：${size}`,`批大小：${batch===-1?'自动（约60%显存）':batch}`,`数据线程：${['official26','custom26'].includes(cfg?.profile)?8:0}`,`混合精度：${['official26','custom26'].includes(cfg?.profile)?'开启':'关闭'}`,`随机种子：${['official26','custom26'].includes(cfg?.profile)?0:cfg?.seed ?? 42}`,`优化器：${cfg?.optimizer || 'auto'}`].map(text=>{const span=document.createElement('span');span.textContent=text;return span;}));
    const auto=$('#training-optimizer').value==='auto';for(const id of ['lr0','momentum'])$('#training-'+id).disabled=auto;
    $('#training-capability-status').textContent=!cfg?(verified()?'沿用当前目标已验证的640 / batch1 / AMP关闭及原运行库默认增强。':'原默认增强方案尚未在当前GPU验证，请选择已有实测的方案。'):verified()?'此方案与优化器组合已有当前目标的成功实例记录；运行前仍会检查环境。':'此方案与优化器组合待当前GPU验证：可编辑和预览，暂不能提交训练。不会自动运行GPU。';
    if(cfg?.profile==='custom26')$('#training-capability-status').textContent='仅覆盖你修改的输入尺寸和批大小，其余沿用官方默认参数。显存不足时会明确提示，不自动调整参数。';
    if(cfg?.profile==='precision')$('#training-capability-status').textContent=verified()?'可调方案已有当前GPU运行证据；尺寸与批大小是否适合本数据，需看验证表现。运行前仍核对环境和显存。':'可调方案尚无当前GPU运行证据，暂不能提交；可使用已验证兼容方案。';
    if(options?.validation_basis==='environment'&&verified())$('#training-capability-status').textContent='已有环境校验通过。实际训练的显存占用取决于模型、输入尺寸和批大小，可按需要调整。';
    showSmallTargets();
    onChange();
  }
  function draw(svg,url,width,height,boxes,classes){
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.append(node('image',{href:url,width,height}));
    boxes.forEach((b,index)=>{const color=['#e76f51','#2a9d8f','#e9c46a','#5a86c5','#9b6bcc','#e07aa8','#8dad45','#cf8a3f','#38a6c1','#7c879c'][(classes.find(c=>c.id===b.class_id)?.index ?? 0)%10];const r=node('rect',{x:b.x,y:b.y,width:b.width,height:b.height,fill:'none',stroke:color,'stroke-width':Math.max(width/300,1)});const title=node('title');title.textContent=`${classes.find(c=>c.id===b.class_id)?.name || b.class_id} · x=${b.x.toFixed(2)}, y=${b.y.toFixed(2)}, w=${b.width.toFixed(2)}, h=${b.height.toFixed(2)}`;r.append(title);svg.append(r);});
  }
  async function preview(next=false){
    const cfg=value();if(!cfg || !context)return;if(['official26','custom26'].includes(cfg.profile)){$('#augmentation-status').textContent='官方默认包含随机组合增强，训练时执行；几何示意可在手动方案中查看。';return;}invalidate();const current=token,target={...context},version=$('#training-version').value;if(next)sample=(sample+1)%10000;
    $('#augmentation-preview').disabled=true;$('#augmentation-next').disabled=true;$('#augmentation-status').textContent='正在核对冻结标签并生成几何示意…';
    try{
      const base=`/api/projects/${target.projectId}/tasks/${target.taskId}`;
      if(!manifest || manifest.version_id!==version){
        const response=await request(`${base}/versions/${version}`);if(current!==token)return;
        manifest={...response.manifest,version_id:version};const select=$('#augmentation-image');select.replaceChildren();
        for(const image of manifest.images){const o=document.createElement('option');o.value=image.id;o.textContent=`${image.subset==='train'?'训练':image.subset==='val'?'验证':'测试'} · ${image.filename}`;select.append(o);}
      }
      const image=manifest.images.find(i=>i.id===$('#augmentation-image').value);
      const data=await request(`${base}/training-preview`,{method:'POST',body:{version_id:version,image_id:image.id,configuration:cfg,sample}});if(current!==token)return;
      draw($('#augmentation-original'),`${base}/versions/${version}/images/${image.id}`,image.width,image.height,image.boxes,data.classes);
      draw($('#augmentation-transformed'),data.image_data,data.geometry.width,data.geometry.height,data.geometry.boxes,data.classes);
      $('#augmentation-summary').textContent=`${data.filename} · ${data.subset} · 示例 ${sample}\n缩放后 ${data.geometry.resized_width} × ${data.geometry.resized_height}；左/上留边 ${data.geometry.left} / ${data.geometry.top}px\n水平翻转：${data.geometry.horizontal?'已应用':'未应用'}；垂直翻转：${data.geometry.vertical?'已应用':'未应用'}\n${data.scope}\n${data.validation_notice}`;
      $('#augmentation-parameters').textContent=JSON.stringify(data.parameters,null,2);$('#augmentation-result').hidden=false;$('#augmentation-status').textContent='预览完成；只读使用冻结版本，未创建训练。';
    }catch(e){if(current===token){$('#augmentation-error').textContent=e.message;$('#augmentation-error').hidden=false;$('#augmentation-status').textContent='预览未完成，可修改或重试。';}}
    finally{if(current===token){$('#augmentation-preview').disabled=false;$('#augmentation-next').disabled=false;}}
  }
  function fill(cfg,epochs){
    $('#training-profile').value=cfg?.profile || 'legacy';$('#training-imgsz').value=cfg?.imgsz||1824;$('#training-batch').value=cfg?.batch??-1;
    for(const id of ids)$('#training-'+id).value=cfg?.[id] ?? ({fliplr:0,flipud:0,seed:42,patience:100,optimizer:'auto',lr0:.001,momentum:.9}[id]);
    if(epochs)$('#training-epochs').value=epochs;update();
  }
  function showRecommendation(){
    const p=recommendation?.parameters;
    $('#training-recommendation').textContent=recommendation?[`推荐：输入${p.imgsz} / ${p.batch===-1?'AutoBatch -1（约60%显存）':'batch'+p.batch} / ${recommendation.epochs}轮`,recommendation.mode==='official'?'使用YOLO26官方默认训练参数。':recommendation.mode==='history'?`优先复用同版本训练「${recommendation.source.training_name}」；这是已有内置验证表现较好的配置。`:'推荐依据：图像、目标尺寸与样本量（经验起点），尚无可比历史。',recommendation.warnings[0] || recommendation.notice,$('#training-parameter-mode').value==='manual'?'当前为手动设置，修改不会被推荐覆盖。':'当前已应用推荐；修改参数可转为手动。'].join('\n'):'正在读取冻结数据并生成推荐…';
    $('#training-recommendation-details').textContent=recommendation?[...recommendation.reasons,...recommendation.warnings,recommendation.notice].join('\n'):'';
    showSmallTargets();
  }
  function showSmallTargets(){
    const data=recommendation?.small_targets;$('#training-small-targets').hidden=!data;
    if(!data)return;const cfg=value(),profile=options?.configuration_profiles.find(p=>p.id===cfg?.profile),size=cfg?.imgsz || profile?.imgsz || 640,current=data.sizes.find(row=>row.imgsz===size) || data.current;
    $('#training-small-summary').textContent=data.target_count?`输入${size}px时：${data.target_count}个训练目标中，框短边不足4px有${current.under4}个，不足8px有${current.under8}个；较小10%目标的框短边约${current.p10_short_pixels}px。细长框${data.thin_box_count}个，无目标训练图${data.negative_images}张。`:'当前训练集没有可分析的正目标；请先补充代表性小目标标注。';
    $('#training-small-scope').textContent=data.scope+(data.size_limit_insufficient?' 按当前标注估算，2048输入仍不足16px；参数调整不能补回采集细节。':'');
    $('#training-small-sizes').replaceChildren(...data.sizes.map(row=>{const tr=document.createElement('tr');for(const value of [row.imgsz,row.p10_short_pixels===null?'无目标':row.p10_short_pixels,row.under4,row.under8]){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;}));
    $('#training-small-guidance').replaceChildren(...data.guidance.map(text=>{const item=document.createElement('li');item.textContent=text;return item;}));
    const button=$('#training-small-apply-size');button.disabled=!data.suggested_imgsz;button.textContent=data.suggested_imgsz?`按目标尺寸设置输入 ${data.suggested_imgsz}（实验起点）`:'暂无尺寸建议';
  }
  $('#training-small-apply-size').onclick=()=>{
    const size=recommendation?.small_targets?.suggested_imgsz;if(!size)return;const cfg=value() || {fliplr:0,flipud:0,seed:42,patience:30,optimizer:'auto'};
    const profile=options?.configuration_profiles.find(p=>p.id===cfg.profile);$('#training-parameter-mode').value='manual';fill({...(options?.default_configuration?.profile==='official26'?{}:cfg),profile:options?.default_configuration?.profile==='official26'?'custom26':'precision',imgsz:size,batch:cfg.batch??profile?.batch??16},Number($('#training-epochs').value));showRecommendation();
  };
  function applyRecommendation(){if(!recommendation)return;$('#training-parameter-mode').value='recommended';fill(recommendation.configuration,recommendation.epochs);showRecommendation();}
  function manual(){if($('#training-parameter-mode').value!=='manual'){$('#training-parameter-mode').value='manual';showRecommendation();}onChange();}
  async function loadRecommendation(){
    if(!context)return;const version=$('#training-version').value,current=++recommendationToken;recommendation=null;pendingRecommendation=true;showRecommendation();onChange();
    try{const result=await request(`/api/training-options?project_id=${context.projectId}&task_id=${context.taskId}&version_id=${version}&training_mode=${$('#training-model-mode').value}`);if(current!==recommendationToken)return;
      recommendation=result.recommendation;pendingRecommendation=false;if($('#training-parameter-mode').value==='recommended')applyRecommendation();else showRecommendation();onChange();
    }catch(e){if(current!==recommendationToken)return;pendingRecommendation=false;$('#training-recommendation').textContent=`推荐暂不可用：${e.message}。可切换手动设置；不会自动启动训练。`;onChange();}
  }
  $('#training-parameter-mode').onchange=()=>{if($('#training-parameter-mode').value==='recommended'){if(recommendation)applyRecommendation();else loadRecommendation();}else{const cfg=value(),profile=options?.configuration_profiles.find(p=>p.id===cfg?.profile);if(cfg && !['precision','custom26'].includes(cfg.profile))fill({profile:'custom26',imgsz:profile.imgsz,batch:profile.batch},Number($('#training-epochs').value));showRecommendation();onChange();}};
  $('#training-recommendation-apply').onclick=()=>recommendation?applyRecommendation():loadRecommendation();
  async function loadModelMode(){
    if(!context)return;const current=++modelToken,mode=$('#training-model-mode').value,old=value(),epochs=Number($('#training-epochs').value),setting=$('#training-parameter-mode').value;
    ++recommendationToken;modelPending=true;pendingRecommendation=false;recommendation=null;update();
    try{
      const result=await request(`/api/training-options?project_id=${context.projectId}&task_id=${context.taskId}&version_id=${$('#training-version').value}&training_mode=${mode}`);
      if(current!==modelToken)return;options=result;modelPending=false;recommendation=result.recommendation;
      for(const p of options.configuration_profiles){const option=$('#training-profile').querySelector(`option[value="${p.id}"]`);if(p.id==='custom26'){option.textContent='自定义 · 基于官方默认参数';continue;}if(p.id==='official26'){option.textContent=`YOLO26 · 官方默认参数 · ${p.enabled?'可用':'待校验'}`;continue;}option.textContent=`${p.id==='precision'?'自定义方案':'几何增强'} · ${p.imgsz} / ${p.batch===-1?'自动批大小':'batch'+p.batch} · ${p.enabled?'可用':'待校验'}`;}
      if(setting==='recommended')applyRecommendation();else fill(old || {profile:'custom26',imgsz:640,batch:16},epochs);
      showRecommendation();update();
      if(!options.training_modes.find(m=>m.id===mode)?.enabled)$('#training-capability-status').textContent='此模式尚未在当前GPU完成验证，可查看配置；通用模式仍可使用。';
    }catch(e){if(current===modelToken){modelPending=false;$('#training-capability-status').textContent=e.message;onChange();}}
  }
  $('#training-model-mode').onchange=loadModelMode;
  for(const selector of ['#training-profile','#training-imgsz','#training-batch','#training-epochs',...ids.map(id=>'#training-'+id)])$(selector).addEventListener('change',manual);
  $('#training-profile').onchange=()=>{if(['precision','custom26'].includes($('#training-profile').value)){const custom=$('#training-profile').value==='custom26';$('#training-imgsz').value=custom?640:1824;$('#training-batch').value=custom?16:-1;}update();};$('#training-imgsz').onchange=update;$('#training-batch').onchange=update;for(const id of ids)$('#training-'+id).addEventListener('change',update);
  $('#augmentation-preview').onclick=()=>preview();$('#augmentation-next').onclick=()=>preview(true);$('#augmentation-image').onchange=()=>preview();
  return {value,verified,pending(){return modelPending || pendingRecommendation;},reset(projectId,taskId,config,previous,preserveLegacy=false){
    ++modelToken;modelPending=false;
    context={projectId,taskId};options=config;manifest=null;sample=0;recommendation=null;recommendationToken++;pendingRecommendation=false;$('#training-parameter-mode').value=preserveLegacy?'manual':'recommended';
    for(const p of options.configuration_profiles){const option=$('#training-profile').querySelector(`option[value="${p.id}"]`);if(p.id==='custom26'){option.textContent='自定义 · 基于官方默认参数';continue;}if(p.id==='official26'){option.textContent=`YOLO26 · 官方默认参数 · ${p.enabled?'可用':'待校验'}`;continue;}option.textContent=`${p.id==='precision'?'自定义方案':'几何增强'} · ${p.imgsz} / ${p.batch===-1?'自动批大小':'batch'+p.batch} · ${p.enabled?'可用':'待校验'}`;}
    $('#training-profile').value=previous?.profile || (preserveLegacy?(options.task_type==='segment'?'geometry640':'legacy'):(options.default_configuration?.profile || 'official26'));$('#training-imgsz').value=previous?.imgsz||1824;$('#training-batch').value=previous?.batch??-1;for(const id of ids)$('#training-'+id).value=previous?.[id] ?? ({fliplr:0,flipud:0,seed:42,patience:100,optimizer:'auto',lr0:.001,momentum:.9}[id]);update();if((options.training_mode || 'standard')!==$('#training-model-mode').value)loadModelMode();
  },invalidate(){invalidate();++modelToken;++recommendationToken;modelPending=false;pendingRecommendation=false;},sync(){for(const id of ['lr0','momentum'])$('#training-'+id).disabled=$('#training-optimizer').value==='auto';},changedVersion(){manifest=null;invalidate();if(modelPending || (options?.training_mode || 'standard')!==$('#training-model-mode').value)loadModelMode();else loadRecommendation();}};
}
