import {request} from './api.js';
const $=s=>document.querySelector(s);
export function createRoiEditor({notify,onSaved}){
 let context,setting,image,region,start,busy=false,token=0;
 const canvas=$('#roi-canvas'),ctx=canvas.getContext('2d'),dialog=$('#roi-dialog');
 function draw(){if(!image)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0,canvas.width,canvas.height);if(region){const [x,y,r,b]=region;ctx.fillStyle='#061f3266';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,x*image.width,y*image.height,(r-x)*image.width,(b-y)*image.height,x*canvas.width,y*canvas.height,(r-x)*canvas.width,(b-y)*canvas.height);ctx.strokeStyle='#14d7a0';ctx.lineWidth=3;ctx.strokeRect(x*canvas.width,y*canvas.height,(r-x)*canvas.width,(b-y)*canvas.height);}$('#roi-status').textContent=region?`区域：左${(region[0]*100).toFixed(1)}% / 上${(region[1]*100).toFixed(1)}%，宽${((region[2]-region[0])*100).toFixed(1)}% / 高${((region[3]-region[1])*100).toFixed(1)}%`:'使用整图';}
 const point=e=>{const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];};
 canvas.onpointerdown=e=>{if(busy)return;start=point(e);canvas.setPointerCapture(e.pointerId);};
 canvas.onpointermove=e=>{if(!start)return;const p=point(e);region=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];draw();};
 canvas.onpointerup=()=>{start=null;};canvas.onpointercancel=()=>{start=null;};
 $('#dataset-roi').onclick=async()=>{
  const current=++token;$('#roi-error').hidden=true;$('#roi-save').disabled=true;
  try{const list=await request(`${context}/images?page_size=1`);if(!list.images.length)throw new Error('请先导入图片。');setting=await request(context+'/roi');region=JSON.parse(setting.region_json);image=new Image();image.onload=()=>{if(current!==token)return;canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;draw();$('#roi-save').disabled=false;};image.onerror=()=>{$('#roi-error').hidden=false;$('#roi-error').textContent='参考图片读取失败，尚未修改ROI。';};image.src=`${context}/images/${list.images[0].id}/original`;dialog.showModal();}catch(e){notify(e.message);}
 };
 $('#roi-clear').onclick=()=>{region=null;draw();};$('#roi-cancel').onclick=()=>{if(!busy){token++;dialog.close();}};
 $('#roi-save').onclick=async()=>{if(busy)return;if(region&&(region[2]-region[0]<1/image.width||region[3]-region[1]<1/image.height)){$('#roi-error').textContent='ROI至少为一个像素，请重新拖动。';$('#roi-error').hidden=false;return;}busy=true;for(const n of dialog.querySelectorAll('button'))n.disabled=true;
  try{await request(context+'/roi',{method:'PUT',body:{revision:setting.revision,region}});dialog.close();notify('数据集ROI已保存，请重新检查与划分后再训练。');await onSaved();}catch(e){$('#roi-error').textContent=e.message;$('#roi-error').hidden=false;}finally{busy=false;for(const n of dialog.querySelectorAll('button'))n.disabled=false;}
 };dialog.addEventListener('cancel',e=>{if(busy)e.preventDefault();});
 return {async open(pid,did){context=`/api/projects/${pid}/datasets/${did}`;setting=await request(context+'/roi');$('#dataset-roi-summary').textContent=JSON.parse(setting.region_json)?'当前已应用数据集统一ROI；标注和后续训练仅使用区域内。':'当前使用整图，可设置统一ROI裁剪区域。';}};
}
