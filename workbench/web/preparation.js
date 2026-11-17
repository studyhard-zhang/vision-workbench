import { request } from './api.js';
const $=selector=>document.querySelector(selector);
const names={train:'训练集',val:'验证集',test:'测试集'};

export function createPreparationWorkspace({ notify, onImage, onSaved, onData=()=>{} }) {
  let context=null,data=null,generation=0,saving=false,page=1;
  const base=ctx=>`/api/projects/${ctx.projectId}/tasks/${ctx.taskId}`;
  function tableRow(values) {
    const tr=document.createElement('tr');for (const value of values) {const td=document.createElement('td');td.textContent=value;tr.append(td);}return tr;
  }
  function classChart(report, split) {
    const chart=$('#preparation-class-chart');chart.replaceChildren();
    const current=split&&!split.stale;
    const totals=report.classes.map(row=>current?Object.values(split.distribution).reduce((sum,subset)=>sum+(subset.classes[row.id]||0),0):row.box_count);
    const maximum=Math.max(1,...totals);
    report.classes.forEach((row,index)=>{
      const item=document.createElement('div');item.className='class-chart-row';
      const label=document.createElement('strong');label.textContent=row.name;
      const track=document.createElement('div');track.className='class-chart-track';
      const counts=current?['train','val','test'].map(subset=>({subset,count:split.distribution[subset].classes[row.id]||0})):[{subset:'pending',count:row.box_count}];
      const description=counts.map(({subset,count})=>`${names[subset]||'待划分'} ${count}`).join(' · ');
      track.setAttribute('role','img');track.setAttribute('aria-label',`${row.name}，${description}，共${totals[index]}个标注目标`);
      for(const {subset,count} of counts){
        if(!count)continue;
        const segment=document.createElement('span');segment.className=`class-chart-segment ${subset}`;segment.style.width=`${count/maximum*100}%`;segment.title=`${names[subset]||'待划分'} ${count} 个`;track.append(segment);
      }
      const number=document.createElement('span');number.className='class-chart-count';number.textContent=`${totals[index]} 个目标`;
      const details=document.createElement('small');details.textContent=description;
      item.append(label,track,number,details);chart.append(item);
    });
    $('#preparation-chart-note').textContent=split?.stale?'当前标注数量，划分已过期，请重新生成以查看各集合分布。':'按标注目标计数，同一张图片可包含多个目标。条形总长度使用统一数量刻度。';
    $('#preparation-chart-legend').hidden=!current;
  }
  function memberList() {
    const filter=$('#split-member-filter').value,q=$('#split-member-search').value.trim().toLowerCase();
    const rows=(data?.split?.images || []).filter(row=>(!filter || row.subset===filter) && `${row.filename}\n${row.group_id}`.toLowerCase().includes(q));
    const pages=Math.max(1,Math.ceil(rows.length/30));page=Math.min(page,pages);
    $('#split-members').replaceChildren();
    for (const row of rows.slice((page-1)*30,page*30)) {
      const tr=tableRow([names[row.subset],row.filename]);
      const td=document.createElement('td'),button=document.createElement('button');button.className='prep-link';button.textContent='查看图片';button.onclick=()=>onImage(row.image_id);td.append(button);tr.append(td);$('#split-members').append(tr);
    }
    $('#split-members-position').textContent=`${page} / ${pages} 页 · ${rows.length} 张`;
    $('#split-members-prev').disabled=page<=1;$('#split-members-next').disabled=page>=pages;
  }
  function render() {
    const {report,split}=data;
    onData(data);
    $('#preparation-status').textContent=report.error_count ? `${report.error_count} 个阻断项，请先处理后再划分。` : `检查通过 · ${report.eligible_count} 张有效图片`;
    $('#preparation-status').dataset.state=report.error_count ? 'error' : 'ok';
    $('#preparation-counts').textContent=`本次参与：已标注 ${report.states.annotated} 张 · 确认无目标 ${report.states.no_target} 张`;
    $('#preparation-issues').replaceChildren();
    for (const item of report.issues) {
      const li=document.createElement('li');li.className=item.severity;
      const text=document.createElement('span');text.textContent=`${item.filename ? item.filename+'：' : ''}${item.message}`;li.append(text);
      if (item.image_id) {const button=document.createElement('button');button.textContent='定位图片';button.className='prep-link';button.onclick=()=>onImage(item.image_id);li.append(button);}
      $('#preparation-issues').append(li);
    }
    $('#preparation-class-stats').replaceChildren();
    for (const row of report.classes) $('#preparation-class-stats').append(tableRow([row.name,row.image_count,row.box_count]));
    classChart(report,split);
    $('#split-generate').disabled=saving || Boolean(report.error_count);
    $('#split-result').hidden=!split;
    if (!split) return;
    $('#split-validity').textContent=split.stale ? '划分已过期：数据、类别、状态或ROI已变化。请重新生成，旧分配仅供参考。' : '划分已保存，可使用下方数据版本创建训练。';
    $('#split-validity').dataset.state=split.stale ? 'error' : 'ok';
    if(!split.stale)$('#split-validity').textContent+=split.algorithm==='class-stratified-v1'?' 当前按缺陷类别分层。':' 当前按图片数量划分。';
    $('#split-distribution').replaceChildren();
    const total=Object.values(split.distribution).reduce((sum,row)=>sum+row.image_count,0);
    for (const subset of ['train','val','test']) {
      const row=split.distribution[subset];$('#split-distribution').append(tableRow([names[subset],split[subset+'_percent']+'%',subset==='test' && !split.test_percent ? '未设置' : row.image_count,`${total ? (row.image_count*100/total).toFixed(1) : 0}%`,row.positive_count,row.no_target_count]));
    }
    $('#split-class-distribution').replaceChildren();
    for (const row of report.classes) $('#split-class-distribution').append(tableRow([row.name,...['train','val','test'].map(s=>split.distribution[s].classes[row.id] || 0)]));
    $('#split-warnings').textContent=split.warnings.join('；');memberList();
  }
  async function open(projectId,taskId) {
    context={projectId,taskId};const target=context,version=++generation;data=null;page=1;
    onData(null);
    $('#preparation-class-chart').replaceChildren();$('#preparation-chart-note').textContent='';$('#preparation-chart-legend').hidden=true;
    $('#preparation-status').textContent='正在检查已标注图片与文件…';$('#preparation-status').dataset.state='';
    $('#preparation-error').hidden=true;$('#preparation-issues').replaceChildren();$('#preparation-counts').textContent='';$('#preparation-class-stats').replaceChildren();$('#split-result').hidden=true;$('#split-generate').disabled=true;
    try {
      const result=await request(`${base(target)}/preparation`);if (version!==generation) return;
      data=result;
      $('#split-strategy').value='classes';
      for (const key of ['train_percent','val_percent','test_percent','seed']) $(`#split-${key}`).value=result.split?.[key] ?? {train_percent:80,val_percent:20,test_percent:0,seed:42}[key];
      render();
    } catch (error) {if (version===generation) {$('#preparation-status').textContent='检查未完成';$('#preparation-error').textContent=error.message;$('#preparation-error').hidden=false;}}
  }
  $('#preparation-refresh').onclick=()=>{if (context && !saving) open(context.projectId,context.taskId);};
  $('#split-form').onsubmit=async event=>{
    event.preventDefault();if (!data || saving || data.report.error_count || !context) return;
    const target=context,body={fingerprint:data.report.fingerprint,revision:data.split?.revision || 0,strategy:$('#split-strategy').value};
    for (const key of ['train_percent','val_percent','test_percent','seed']) {
      const value=$(`#split-${key}`).value;
      if (!/^\d+$/.test(value)) {$('#preparation-error').textContent='比例和种子均须为非负整数。';$('#preparation-error').hidden=false;return;}
      body[key]=Number(value);
    }
    if (body.train_percent<1 || body.val_percent<1 || body.train_percent+body.val_percent+body.test_percent!==100) {$('#preparation-error').textContent='训练和验证比例须大于 0；三项比例合计须为 100%。';$('#preparation-error').hidden=false;return;}
    saving=true;$('#preparation-error').hidden=true;$('#split-generate').textContent='正在检查并保存…';
    for (const node of $('#split-form').querySelectorAll('button,input,select')) node.disabled=true;
    $('#preparation-refresh').disabled=true;
    try {
      data=await request(`${base(target)}/split`,{method:'POST',body});render();
      notify(`划分已完成，已生成 V${String(data.auto_version.version.number).padStart(3,'0')}`);
      await onSaved();
    } catch (error) {try{data=await request(`${base(target)}/preparation`);render();}catch{}$('#preparation-error').textContent=error.message;$('#preparation-error').hidden=false;}
    finally {saving=false;for (const node of $('#split-form').querySelectorAll('button,input,select')) node.disabled=false;$('#preparation-refresh').disabled=false;$('#split-generate').textContent='划分并生成版本';$('#split-generate').disabled=Boolean(data?.report.error_count);}
  };
  $('#split-member-filter').onchange=()=>{page=1;memberList();};$('#split-member-search').oninput=()=>{page=1;memberList();};
  $('#split-members-prev').onclick=()=>{page--;memberList();};$('#split-members-next').onclick=()=>{page++;memberList();};
  return {open,get busy(){return saving;},hide(){generation++;context=null;data=null;}};
}
