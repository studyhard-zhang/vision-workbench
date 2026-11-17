import { request } from './api.js';
import { chartGeometry,formatMetric } from './training-chart.mjs';
const $=s=>document.querySelector(s);
const palette=['#087f8c','#b45309','#6b4dc4','#bc3971'];
const groups=[['box','Box 损失 · 训练 / 验证'],['cls','分类损失 · 训练 / 验证'],['dfl','DFL 损失 · 训练 / 验证'],['metrics','训练内置验证指标'],['lr','逐轮学习率']];
const valueText=value=>typeof value==='string'?value:JSON.stringify(value);
function cell(row,text){const cell=document.createElement('td');cell.textContent=text;row.append(cell);return cell;}
function svgNode(name,attrs={},text){
  const node=document.createElementNS('http://www.w3.org/2000/svg',name);
  for(const [key,value] of Object.entries(attrs))node.setAttribute(key,String(value));
  if(text!==undefined)node.textContent=text;return node;
}

function renderChart(data,group,title,epoch){
  const card=document.createElement('section');card.className='training-chart-card';
  const heading=document.createElement('h4');heading.textContent=title;card.append(heading);
  const series=data.series.filter(s=>s.group===group && s.present);
  const available=series.some(s=>data.rows.some(row=>row[s.key]!==null));
  if(!available){const empty=document.createElement('p');empty.className='training-help';empty.textContent='本次训练未记录这些指标。';card.append(empty);return card;}
  const layout=chartGeometry(data.rows,series.map(s=>s.key),['metrics','mask_metrics'].includes(group));
  const svg=svgNode('svg',{viewBox:'0 0 640 248',role:'img','aria-label':`${title}，第${layout.first}至${layout.last}轮。详细数值见下方逐轮表格。`});
  svg.append(svgNode('title',{},title));
  for(const tick of layout.ticks){
    svg.append(svgNode('line',{x1:layout.left,x2:layout.right,y1:tick.y,y2:tick.y,stroke:'#e5e9ee'}));
    svg.append(svgNode('text',{x:layout.left-7,y:tick.y+4,'text-anchor':'end'},formatMetric(tick.value,true)));
  }
  for(const n of [...new Set([layout.first,Math.round((layout.first+layout.last)/2),layout.last])]){
    const x=layout.first===layout.last?(layout.left+layout.right)/2:layout.left+(n-layout.first)/(layout.last-layout.first)*(layout.right-layout.left);
    svg.append(svgNode('text',{x,y:232,'text-anchor':'middle'},`${n} 轮`));
  }
  const selected=data.rows.find(r=>r.epoch===epoch);
  if(selected){
    const x=layout.first===layout.last?(layout.left+layout.right)/2:layout.left+(epoch-layout.first)/(layout.last-layout.first)*(layout.right-layout.left);
    svg.append(svgNode('line',{x1:x,x2:x,y1:layout.top,y2:layout.bottom,stroke:'#94a3b8','stroke-dasharray':'4 4'}));
  }
  layout.series.forEach((item,index)=>{
    for(const segment of item.segments){
      if(segment.length>1)svg.append(svgNode('polyline',{points:segment.map(p=>`${p.x},${p.y}`).join(' '),fill:'none',stroke:palette[index%palette.length],'stroke-width':2,'stroke-dasharray':index===1 && group!=='metrics'?'5 3':'none'}));
      for(const point of segment){
        const dot=svgNode('circle',{cx:point.x,cy:point.y,r:point.epoch===epoch?4:2,fill:palette[index%palette.length]});
        dot.append(svgNode('title',{},`第 ${point.epoch} 轮 · ${series[index].label}：${formatMetric(point.value)}`));svg.append(dot);
      }
    }
  });
  card.append(svg);const legend=document.createElement('ul');legend.className='training-chart-legend';
  series.forEach((s,index)=>{const item=document.createElement('li'),mark=document.createElement('span');mark.style.backgroundColor=palette[index%palette.length];mark.setAttribute('aria-hidden','true');item.append(mark,document.createTextNode(`${s.label}：${formatMetric(selected?.[s.key])}`));legend.append(item);});
  card.append(legend);return card;
}

export function createTrainingReport(){
  let token=0,controller=null,context=null,data=null;
  const charts=$('#training-report-charts'),status=$('#training-report-status'),issue=$('#training-report-error');
  function clear(){
    for(const id of ['precision','recall','map'])$('#training-key-'+id).textContent='—';
    $('#training-key-context').textContent='正在读取训练内置验证指标…';$('#training-key-map50').textContent='mAP@0.5–0.95：未记录';
    $('#training-key-diagnosis').textContent='';data=null;charts.replaceChildren();$('#training-report-row-head').replaceChildren();$('#training-report-rows').replaceChildren();
    $('#training-report-parameters').replaceChildren();$('#training-report-warnings').replaceChildren();
    $('#training-report-controls').hidden=true;$('#training-report-data').hidden=true;$('#training-report-parameter-panel').hidden=true;
    $('#training-report-source').textContent='';$('#training-report-optimizer').textContent='';issue.hidden=true;issue.textContent='';
  }
  function draw(){
    charts.replaceChildren();const epoch=Number($('#training-report-epoch').value);
    const row=data.rows.find(r=>r.epoch===epoch),suffix=data.task_type==='segment'?'M':'B';
    for(const [id,key] of [['precision','precision'],['recall','recall'],['map','mAP50']]){const v=row?.[`metrics/${key}(${suffix})`];$('#training-key-'+id).textContent=typeof v==='number'?`${(v*100).toFixed(2)}%`:'未记录';}
    const map5095=row?.[`metrics/mAP50-95(${suffix})`];$('#training-key-map50').textContent=`mAP@0.5–0.95：${typeof map5095==='number'?`${(map5095*100).toFixed(2)}%`:'未记录'}`;
    $('#training-key-diagnosis').textContent=(data.diagnosis?.notes || []).join('\n');
    $('#training-key-context').textContent=`第 ${epoch} 轮${epoch===data.rows.at(-1)?.epoch?'（最后记录轮次）':''} · ${suffix==='M'?'分割掩膜':'检测框'} · 训练内置验证集，非独立测试；不代表 best.pt 指标。`;
    for(const [group,title] of [...groups,...(data.task_type==='segment'?[['seg','分割掩膜损失 · 训练 / 验证'],['mask_metrics','分割掩膜验证指标']]:[])])charts.append(renderChart(data,group,title,epoch));
    $('#training-report-selected').textContent=`当前查看第 ${epoch} 轮；图例数值与原始逐轮记录对应。`;
    for(const row of $('#training-report-rows').children)row.classList.toggle('selected-epoch',Number(row.dataset.epoch)===epoch);
  }
  function render(){
    status.textContent=data.status==='AVAILABLE'?`已记录 ${data.rows.length} 轮 · 请求 ${data.requested_epochs} 轮 · 回传完成 ${data.completed_epochs ?? '未知'} 轮`:data.reason;
    $('#training-report-parameter-panel').hidden=false;
    $('#training-report-optimizer').textContent=data.actual_optimizer?`最终优化器：${data.actual_optimizer}。实际解析参数是 trainer.args 的回传值；auto 下的 lr0 / momentum 配置不等于优化器最终值，逐轮学习率以 lr/pg* 为准。`:'实际参数与最终优化器尚未回传。请求参数已冻结，未传回项不会用默认值补齐。';
    for(const parameter of data.parameters){
      const row=document.createElement('tr');cell(row,parameter.key);
      cell(row,parameter.requested_present?valueText(parameter.requested):'未显式指定');
      cell(row,parameter.actual_present?valueText(parameter.actual):'未回传');cell(row,parameter.note || (parameter.requested_present && parameter.actual_present && JSON.stringify(parameter.requested)!==JSON.stringify(parameter.actual)?'解析值与请求不同，保留两者。':''));
      $('#training-report-parameters').append(row);
    }
    for(const warning of data.warnings){const item=document.createElement('li');item.textContent=warning;$('#training-report-warnings').append(item);}
    if(data.status!=='AVAILABLE'){$('#training-key-context').textContent=data.reason;return;}
    $('#training-report-controls').hidden=false;$('#training-report-data').hidden=false;
    const select=$('#training-report-epoch');select.replaceChildren();
    for(const row of data.rows){const option=document.createElement('option');option.value=row.epoch;option.textContent=`第 ${row.epoch} 轮`;select.append(option);}select.value=data.rows.at(-1).epoch;
    const present=data.series.filter(s=>s.present),head=$('#training-report-row-head');
    for(const text of ['轮次','累计秒',...present.map(s=>s.label)]){const th=document.createElement('th');th.textContent=text;th.scope='col';head.append(th);}
    for(const observation of data.rows){
      const row=document.createElement('tr');row.dataset.epoch=observation.epoch;
      cell(row,observation.epoch);cell(row,formatMetric(observation.time));for(const s of present)cell(row,formatMetric(observation[s.key]));$('#training-report-rows').append(row);
    }
    $('#training-report-source').textContent=`来源：results.csv · SHA-256 ${data.csv_sha256}。图表显示逐轮记录，末轮数值不等于 best.pt 的最佳值。`;
    draw();
  }
  async function load(){
    const current=++token;controller?.abort();controller=new AbortController();clear();status.textContent='正在核对并读取已回传的训练指标…';$('#training-report-retry').disabled=true;
    try{const response=await request(context.url,{signal:controller.signal});if(current!==token)return;data=response;render();}
    catch(error){if(current!==token || error.name==='AbortError')return;$('#training-key-context').textContent='指标读取失败，可点击“重新读取指标”重试。';status.textContent='训练指标暂不可显示，训练记录和模型状态保持不变。';issue.textContent=error.message;issue.hidden=false;}
    finally{if(current===token)$('#training-report-retry').disabled=false;}
  }
  $('#training-report-epoch').onchange=draw;$('#training-report-retry').onclick=()=>context && load();
  return {
    open(projectId,taskId,job){
      const next={id:`${projectId}/${taskId}/${job.id}/${job.dispatch?.result?.report.status || ''}`,url:`/api/projects/${projectId}/tasks/${taskId}/training/${job.id}/report`};
      if(context?.id===next.id)return;context=next;load();
    },
    close(){token++;controller?.abort();context=null;clear();}
  };
}
