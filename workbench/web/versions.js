import { request } from './api.js';
const $=selector=>document.querySelector(selector);
const subsets={train:'训练集',val:'验证集',test:'测试集'};
const time=value=>new Date(value).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false});
const label=number=>'V'+String(number).padStart(3,'0');

export function createVersionWorkspace({notify,onPublished}) {
  let context=null,preparation=null,rows=[],epoch=0,detailEpoch=0,publishing=false,detail=null,page=1,auditing=false,auditReport=null;
  const dialog=$('#version-dialog');
  const base=ctx=>`/api/projects/${ctx.projectId}/tasks/${ctx.taskId}/versions`;
  function row(values) {
    const tr=document.createElement('tr');
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;
  }
  function controls(){
    const ready=preparation && !preparation.report.error_count && preparation.split && !preparation.split.stale;
    $('#version-publish').disabled=publishing || !ready;
    $('#version-publish').textContent=publishing?'正在发布…':'发布数据版本';
    $('#version-refresh').disabled=publishing;
    $('#version-ready').textContent=ready?'当前划分已保存，版本由划分操作自动生成；重复操作复用同一版本。':'完成上方有效数据划分后，会自动生成数据版本。';
  }
  function render(){
    $('#version-list').replaceChildren();$('#version-empty').hidden=Boolean(rows.length);
    for(const item of rows){
      const card=document.createElement('div');card.className='version-row';
      const heading=document.createElement('strong');heading.textContent=label(item.number);
      const info=document.createElement('span');info.textContent=`${item.image_count} 张 · ${item.class_count} 类 · ${time(item.created_at)}`;
      const button=document.createElement('button');button.className='button secondary';button.textContent='查看数据';
      button.setAttribute('aria-label',`查看数据版本 ${label(item.number)}`);button.onclick=()=>openDetail(item.id);
      card.append(heading,info,button);$('#version-list').append(card);
    }
  }
  async function reload(target=context,version=epoch){
    try{
      const result=await request(base(target));if(version!==epoch)return;
      rows=result.versions;render();
    }catch(error){if(version===epoch){$('#version-error').textContent=error.message;$('#version-error').hidden=false;}}
  }
  async function open(projectId,taskId){
    context={projectId,taskId};preparation=null;rows=[];const version=++epoch;
    $('#version-error').hidden=true;render();controls();await reload(context,version);
  }
  function members(){
    const filter=$('#version-subset').value;
    const images=detail.manifest.images.filter(image=>!filter || image.subset===filter);
    const pages=Math.max(1,Math.ceil(images.length/30));page=Math.min(page,pages);$('#version-members').replaceChildren();
    for(const image of images.slice((page-1)*30,page*30)){
      const tr=row([subsets[image.subset],image.filename,image.state==='no_target'?'确认无目标':`${image.boxes.length} 个${detail.manifest.label_format==='yolo-segment-polygon-normalized-v1'?'多边形':'框'}`]);
      const td=document.createElement('td'),button=document.createElement('button');button.className='prep-link';button.textContent='查看冻结标注';
      button.onclick=()=>preview(image);td.append(button);tr.append(td);$('#version-members').append(tr);
    }
    $('#version-page').textContent=`${page} / ${pages} 页 · ${images.length} 张`;
    $('#version-prev').disabled=page<=1;$('#version-next').disabled=page>=pages;
  }
  function preview(image){
    const manifest=detail.manifest;$('#version-image-panel').hidden=false;
    $('#version-image-name').textContent=`${image.filename} · ${image.width} × ${image.height} · ${subsets[image.subset]}`;
    const svg=$('#version-image');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${image.width} ${image.height}`);
    const element=(name,attributes)=>{const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));return node;};
    const picture=element('image',{href:`${base(context)}/${manifest.version_id}/images/${image.id}`,width:image.width,height:image.height});
    picture.addEventListener('error',()=>{$('#version-label').textContent='训练图片无法读取；请重新检查版本完整性。';});svg.append(picture);
    const indices=new Map(manifest.classes.map(c=>[c.id,c]));const lines=[];
    for(const box of image.boxes){
      const category=indices.get(box.class_id);
      svg.append(box.points?element('polygon',{points:box.points.map(p=>p.join(',')).join(' '),fill:'#ff713f33',stroke:'#ff713f','stroke-width':Math.max(image.width/400,1)}):element('rect',{x:box.x,y:box.y,width:box.width,height:box.height,fill:'none',stroke:'#ff713f','stroke-width':Math.max(image.width/400,1)}));
      const text=element('text',{x:box.x,y:Math.max(box.y-4,12),'font-size':Math.max(image.width/50,12),fill:'#c34714'});text.textContent=category.name;svg.append(text);
      lines.push(`${category.index} ${(box.points?box.points.flatMap(([x,y])=>[x/image.width,y/image.height]):[(box.x+box.width/2)/image.width,(box.y+box.height/2)/image.height,box.width/image.width,box.height/image.height]).map(v=>Number(v.toPrecision(12))).join(' ')}`);
    }
    $('#version-label').textContent=lines.join('\n') || '空标签文件（已明确确认无目标）';
  }
  async function openDetail(id){
    const target=context,version=++detailEpoch;detail=null;page=1;
    $('#version-dialog-title').textContent='正在读取版本…';$('#version-detail-content').hidden=true;$('#version-detail-error').hidden=true;
    $('#version-image-panel').hidden=true;$('#version-image').replaceChildren();dialog.showModal();
    auditReport=null;$('#version-audit-result').hidden=true;$('#version-audit-checks').replaceChildren();$('#version-audit-download').hidden=true;
    $('#version-audit').disabled=true;$('#version-audit').textContent='核对训练数据';
    try{
      const result=await request(`${base(target)}/${id}`);if(version!==detailEpoch)return;
      detail=result;const {manifest,integrity}=result;
      $('#version-dialog-title').textContent=`数据版本 ${label(manifest.number)}`;
      $('#version-detail-meta').textContent=`${manifest.task_name} · ${time(manifest.created_at)} · 划分修订 ${manifest.split.revision} · 种子 ${manifest.split.seed}`;
      $('#version-integrity').textContent=integrity.ok?'版本文件与引用图片校验通过':'版本完整性异常：'+integrity.issues.join('；');
      $('#version-integrity').dataset.state=integrity.ok?'ok':'error';
      $('#version-download').hidden=!integrity.ok;$('#version-download').href=`${base(target)}/${id}/archive`;
      $('#version-audit').disabled=!integrity.ok;
      $('#version-download').download=`data-${label(manifest.number)}.zip`;
      $('#version-class-map').replaceChildren();
      for(const c of manifest.classes)$('#version-class-map').append(row([c.index,c.name,...['train','val','test'].map(s=>manifest.distribution[s].classes[c.id] || 0)]));
      $('#version-distribution').replaceChildren();
      for(const s of ['train','val','test']){const stat=manifest.distribution[s];$('#version-distribution').append(row([subsets[s],s==='test' && !manifest.split.test_percent?'未设置':stat.image_count,stat.positive_count,stat.no_target_count]));}
      $('#version-warnings').textContent=manifest.warnings.join('；');$('#version-subset').value='';members();$('#version-detail-content').hidden=false;
    }catch(error){if(version===detailEpoch){$('#version-dialog-title').textContent='无法读取版本';$('#version-detail-error').textContent=error.message;$('#version-detail-error').hidden=false;}}
  }
  $('#version-publish').onclick=async()=>{
    if(publishing || !context || !preparation?.split || preparation.split.stale || preparation.report.error_count)return;
    const target=context,version=epoch,body={fingerprint:preparation.report.fingerprint,split_revision:preparation.split.revision};
    publishing=true;controls();$('#version-error').hidden=true;
    try{
      const result=await request(base(target),{method:'POST',body});if(version!==epoch)return;
      notify(result.reused?`已发布过，打开已有版本 ${label(result.version.number)}`:`数据版本 ${label(result.version.number)} 已发布`);
      await reload(target,version);await onPublished();if(version===epoch)await openDetail(result.version.id);
    }catch(error){if(version===epoch){$('#version-error').textContent=error.message;$('#version-error').hidden=false;}}
    finally{publishing=false;controls();}
  };
  $('#version-refresh').onclick=()=>{if(context && !publishing){$('#version-error').hidden=true;reload();}};
  $('#version-audit').onclick=async()=>{
    if(!context || !detail || auditing)return;
    const target=context,id=detail.version.id,version=detailEpoch;
    auditing=true;auditReport=null;$('#version-audit').disabled=true;$('#version-audit').textContent='正在核对…';
    $('#version-dialog-close').disabled=true;$('#version-audit-download').hidden=true;
    $('#version-audit-result').hidden=false;$('#version-audit-status').textContent='正在核对该版本的训练图片、标签和划分…';$('#version-audit-status').dataset.state='';$('#version-audit-checks').replaceChildren();
    try{
      const report=await request(`${base(target)}/${id}/readiness`);if(version!==detailEpoch)return;
      auditReport=report;
      if(report.passed && report.source_integrity_checked){$('#version-integrity').textContent='版本文件与引用图片校验通过';$('#version-integrity').dataset.state='ok';$('#version-download').hidden=false;}
      $('#version-audit-status').textContent=report.passed?`数据核对通过 · ${report.stats.image_count} 张图片 · ${report.stats.box_count} 个框 · ${report.total_elapsed_seconds} 秒；尚未运行训练。`:'数据核对未通过，请查看下方原因。';
      $('#version-audit-status').dataset.state=report.passed?'ok':'error';
      for(const check of report.checks){const li=document.createElement('li');li.textContent=`${check.status==='passed'?'通过':'未通过'}：${check.message}`;$('#version-audit-checks').append(li);}
      $('#version-audit-download').hidden=false;
    }catch(error){if(version===detailEpoch){$('#version-audit-status').textContent=error.message;$('#version-audit-status').dataset.state='error';if(error.code==='VERSION_DAMAGED'){$('#version-integrity').textContent=error.message;$('#version-integrity').dataset.state='error';$('#version-download').hidden=true;}}}
    finally{auditing=false;$('#version-dialog-close').disabled=false;if(version===detailEpoch){$('#version-audit').disabled=false;$('#version-audit').textContent='重新核对训练数据';}}
  };
  $('#version-audit-download').onclick=()=>{
    if(!auditReport)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(auditReport,null,2)+'\n'],{type:'application/json;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download=`data-check-${label(auditReport.number)}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
  };
  $('#version-dialog-close').onclick=()=>{if(!auditing){detailEpoch++;dialog.close();}};
  dialog.addEventListener('cancel',event=>{if(auditing)event.preventDefault();else detailEpoch++;});
  $('#version-subset').onchange=()=>{page=1;members();};$('#version-prev').onclick=()=>{page--;members();};$('#version-next').onclick=()=>{page++;members();};
  return {open,get busy(){return publishing || auditing;},setPreparation(value){preparation=value;controls();},hide(){epoch++;detailEpoch++;context=null;preparation=null;detail=null;if(dialog.open)dialog.close();}};
}
