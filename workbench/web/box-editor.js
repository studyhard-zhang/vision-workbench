import { request } from './api.js';
import { BoxDraft, clamp, drawBox, moveBox, resizeBox, handles, hitBox, screenToImage, imageToScreen } from './box-model.mjs';
import { DrawingClassPreference, saveBeforeNext } from './editor-flow.mjs';
import {PolygonDraft,polygonBox,movePolygon,hitPolygon} from './polygon-model.mjs';
import {MIN_SCALE,MAX_SCALE,fitScale,centeredView,zoomAt,focusView} from './editor-view.mjs';
const $ = selector => document.querySelector(selector);

export function createBoxEditor({ beforeLeave, confirmClear, notify }) {
  const dialog=$('#task-image-preview'), canvas=$('#annotation-canvas'), ctx=canvas.getContext('2d');
  let current=null, pending=null, draft=null, annotation=null, image=null, classes=[];
  let generation=0, loading=false, busy=false, navigating=false, changed=false, selected=-1, tool='draw', gesture=null, space=false;
  let polygonPoints=[],polygonClass=null,pointer=null;
  const segmentation=()=>current?.task.task_type==='segment';
  let preferenceStorage=null;
  try {preferenceStorage=window.localStorage;} catch { /* Preferences still work in memory. */ }
  const drawingClass=new DrawingClassPreference(preferenceStorage);
  let zoom=1, pan={x:0,y:0}, size={width:800,height:500}, view={x:0,y:0,scale:1};
  const endpoint = value => `/api/projects/${value.projectId}/tasks/${value.task.id}/images/${value.record.id}/annotation`;
  const isDirty = () => Boolean(draft?.dirty || polygonPoints.length || gesture && gesture.type!=='pan');
  const setError = message => { $('#annotation-error').textContent=message; $('#annotation-error').hidden=!message; };

  function fitView() {
    if (!image || !current) return;
    view=centeredView(current.record,size,zoom,pan);
    $('#canvas-zoom').textContent=`${Number((zoom*100).toFixed(1))}%`;
    $('#canvas-zoom-preset').value=[...$('#canvas-zoom-preset').options].some(option=>Number(option.value)===zoom)?String(zoom):'';
    $('#canvas-zoom-in').disabled=loading||busy||navigating||zoom>=MAX_SCALE;
    $('#canvas-zoom-out').disabled=loading||busy||navigating||zoom<=MIN_SCALE;
  }
  function closeReady(){
    if(!segmentation()||tool!=='draw'||space||gesture||!pointer||polygonPoints.length<3)return false;
    const first=imageToScreen({x:polygonPoints[0][0],y:polygonPoints[0][1]},view);
    return Math.hypot(first.x-pointer.x,first.y-pointer.y)<=10;
  }
  function pointerHint(){
    const closing=closeReady(),panning=space||tool==='pan'||gesture?.type==='pan';
    canvas.dataset.tool=panning?'pan':tool;canvas.dataset.dragging=String(gesture?.type==='pan');canvas.dataset.closeReady=String(closing);
    canvas.title=closing?'点击起点，闭合当前多边形':panning?'拖动平移图片，保留未闭合的顶点':'绘制途中可按住空格或鼠标中键拖动图片；滚轮缩放';
    const text=closing?'已回到起点，点击即可闭合多边形。':polygonPoints.length?
      `正在绘制：${polygonPoints.length} 个顶点。${panning?'平移会保留全部顶点；松开空格/中键，或按D返回绘制。':'点击首点或Enter闭合；空格/中键拖动图片，滚轮缩放；Backspace退一点，Esc取消。'}`:
      '沿目标边缘逐点点击，至少3点后点首点或按Enter闭合。空格/中键拖动图片，滚轮缩放。';
    if($('#annotation-polygon-help').textContent!==text)$('#annotation-polygon-help').textContent=text;
  }
  function renderCanvas() {
    pointerHint();
    const dpr=window.devicePixelRatio || 1;
    ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,size.width,size.height);
    ctx.fillStyle='#192e27';ctx.fillRect(0,0,size.width,size.height);
    if (!image || !current || !draft) return;
    fitView(); const {scale,x,y}=view;
    pointerHint();
    ctx.imageSmoothingEnabled=scale<4;
    ctx.drawImage(image,x,y,current.record.width*scale,current.record.height*scale);
    const boxes=draft.boxes.map((box,index)=>gesture?.index===index && gesture.box ? gesture.box : box);
    if (gesture?.type==='draw' && gesture.box) boxes.push(gesture.box);
    boxes.forEach((box,index)=>{
      const item=classes.find(c=>c.id===box.class_id), color=item?.color || '#f0b678';
      const p=imageToScreen(box,view), w=box.width*scale, h=box.height*scale;
      if(p.x+w<0 || p.y+h<0 || p.x>size.width || p.y>size.height)return;
      ctx.strokeStyle=color;ctx.lineWidth=index===selected ? 2.5 : 1.7;
      if (index===selected && !segmentation()) {ctx.fillStyle=color+'22';ctx.fillRect(p.x,p.y,w,h);}
      if(segmentation()){
        ctx.beginPath();box.points.forEach(([px,py],i)=>i?ctx.lineTo(x+px*scale,y+py*scale):ctx.moveTo(x+px*scale,y+py*scale));ctx.closePath();ctx.fillStyle=color+'44';ctx.fill();ctx.stroke();
      }else ctx.strokeRect(p.x,p.y,w,h);
      const label=`${index+1} ${item?.name || '类别失效'}`;
      ctx.font='12px sans-serif';const tw=Math.min(ctx.measureText(label).width+12,260);
      const labelY=clamp(p.y-23,0,size.height-23),labelX=clamp(p.x,0,Math.max(0,size.width-tw));
      ctx.fillStyle=color;ctx.fillRect(labelX,labelY,tw,21);
      const rgb=[1,3,5].map(start=>parseInt(color.slice(start,start+2),16));
      ctx.fillStyle=rgb[0]*0.299+rgb[1]*0.587+rgb[2]*0.114>145 ? '#152d23' : '#fff';ctx.fillText(label,labelX+5,labelY+15,tw-8);
      if (index===selected && tool==='select') for (const point of (segmentation()?box.points.map(([x,y])=>({x,y})):Object.values(handles(box)))) {
        const hp=imageToScreen(point,view);ctx.fillStyle='#fff';ctx.fillRect(hp.x-4,hp.y-4,8,8);ctx.strokeStyle=color;ctx.strokeRect(hp.x-4,hp.y-4,8,8);
      }
    });
    if(segmentation()&&polygonPoints.length){ctx.beginPath();polygonPoints.forEach(([px,py],i)=>{const p=imageToScreen({x:px,y:py},view);i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);});ctx.strokeStyle=classes.find(c=>c.id===polygonClass)?.color||'#fff';ctx.lineWidth=2;ctx.stroke();for(const [px,py] of polygonPoints){const p=imageToScreen({x:px,y:py},view);ctx.fillStyle='#fff';ctx.fillRect(p.x-4,p.y-4,8,8);}}
    if(segmentation()&&polygonPoints.length){
      const first=imageToScreen({x:polygonPoints[0][0],y:polygonPoints[0][1]},view),last=polygonPoints.at(-1),closing=closeReady();
      if(pointer&&tool==='draw'&&!space&&!gesture){const end=closing?first:pointer;ctx.beginPath();ctx.moveTo(x+last[0]*scale,y+last[1]*scale);ctx.lineTo(end.x,end.y);ctx.strokeStyle=closing?'#5ef0ab':'#ffffffaa';ctx.setLineDash([5,4]);ctx.lineWidth=2;ctx.stroke();ctx.setLineDash([]);}
      ctx.beginPath();ctx.arc(first.x,first.y,closing?11:6,0,Math.PI*2);ctx.fillStyle=closing?'#087f70':'#fff';ctx.fill();ctx.lineWidth=2;ctx.strokeStyle=closing?'#b4ffda':'#087f70';ctx.stroke();
      ctx.font='12px sans-serif';ctx.fillStyle='#fff';const text=closing?'点击闭合':'起点',tx=clamp(first.x+15,3,Math.max(3,size.width-65)),ty=clamp(first.y-15,18,size.height-4);ctx.fillStyle='#163d31';ctx.fillRect(tx-3,ty-14,closing?60:32,20);ctx.fillStyle='#fff';ctx.fillText(text,tx,ty);
    }
    if (draft.state==='excluded') {ctx.fillStyle='#172920aa';ctx.fillRect(0,0,size.width,35);ctx.fillStyle='#f4d0a2';ctx.font='13px sans-serif';ctx.fillText('此图片已排除；可保留框，恢复状态后再标注。',15,23);}
  }
  function resizeCanvas() {
    const rect=canvas.getBoundingClientRect(); if (rect.width<1 || rect.height<1) return;
    size={width:rect.width,height:rect.height};const dpr=window.devicePixelRatio || 1;
    canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);renderCanvas();
  }
  new ResizeObserver(resizeCanvas).observe(canvas);

  function setTool(value) {if(polygonPoints.length&&value==='select'){setError('请先完成或取消当前多边形，再编辑已有标注；可用平移工具移动图片。');return;}tool=value;setError('');render();canvas.focus();}
  function render() {
    const disabled=!draft || loading || busy || navigating;
    for (const button of dialog.querySelectorAll('[data-tool]')) {button.disabled=disabled;button.setAttribute('aria-pressed',String(tool===button.dataset.tool));}
    canvas.dataset.tool=space ? 'pan' : tool;
    $('#draw-class').disabled=disabled || !classes.length;
    $('#annotation-state').disabled=disabled;
    $('#annotation-save').disabled=disabled || !draft?.dirty || Boolean(gesture) || !!polygonPoints.length;
    $('#polygon-finish').hidden=!segmentation();$('#polygon-finish').disabled=disabled||polygonPoints.length<3;
    $('#polygon-cancel').hidden=!segmentation();$('#polygon-cancel').disabled=disabled||!polygonPoints.length;
    $('#annotation-polygon-help').hidden=!segmentation();
    $('#canvas-undo').disabled=disabled || (!draft?.past.length&&!polygonPoints.length);$('#canvas-redo').disabled=disabled || !draft?.future.length;
    $('#box-delete').disabled=disabled || selected<0;
    $('#selected-class').disabled=disabled || selected<0;
    for (const id of ['canvas-zoom-in','canvas-zoom-out','canvas-fit','canvas-actual-size','canvas-zoom-preset']) $(`#${id}`).disabled=disabled;
    $('#canvas-focus-selected').disabled=disabled||selected<0||Boolean(polygonPoints.length);
    const position=current?.navigation?.position?.(current.record.id);
    $('#annotation-prev').disabled=disabled || !position || position.index<=1;
    $('#annotation-next').disabled=disabled || !position || position.index>=position.total;
    $('#annotation-position').textContent=position ? `${position.index} / ${position.total} 张（当前筛选）` : '';
    if (draft) {
      $('#annotation-state').value=draft.state;
      $('#annotation-state option[value="annotated"]').disabled=!draft.boxes.length;
      $('#annotation-feedback').textContent=busy ? '正在保存…' : polygonPoints.length ? '多边形未闭合 · 顶点已保留' : draft.dirty ? '未保存' : `已保存 · 修订 ${annotation.revision}`;
      $('#box-total').textContent=`${draft.boxes.length} / 200 个${segmentation()?'多边形':'框'}`;
    }
    $('#box-list').replaceChildren();
    if (draft) draft.boxes.forEach((box,index)=>{
      const item=classes.find(c=>c.id===box.class_id), button=document.createElement('button');
      button.className='box-list-item';button.textContent=`${index+1} · ${item?.name || '类别失效'}`;
      button.setAttribute('aria-label',`选择${segmentation()?'多边形':'矩形框'} ${index+1}：${item?.name || '类别失效'}`);button.setAttribute('aria-pressed',String(selected===index));button.disabled=disabled;
      button.onclick=()=>{ selected=index;setTool('select'); };$('#box-list').append(button);
    });
    $('#box-empty').hidden=Boolean(draft?.boxes.length);
    $('#box-empty').textContent=classes.length ? segmentation()?'选择绘制类别，沿目标轮廓逐点点击并闭合。':'选择绘制类别，在图片上按住并拖动鼠标画框。' : '请先关闭标注页，在任务内添加类别。无目标和排除状态仍可保存。';
    $('#selected-box').hidden=selected<0 || !draft?.boxes[selected];
    if (selected>=0 && draft?.boxes[selected]) {
      const box=draft.boxes[selected];$('#selected-class').value=box.class_id;
      $('#box-coordinates').textContent=`x ${box.x.toFixed(1)} · y ${box.y.toFixed(1)}\n宽 ${box.width.toFixed(1)} · 高 ${box.height.toFixed(1)} 像素`;
    }
    renderCanvas();
  }
  function resetView() { if(!current)return;zoom=fitScale(current.record,size);pan={x:0,y:0};renderCanvas(); }
  function setZoom(value,point) {
    if (!draft || loading || busy || navigating || gesture) return;
    ({scale:zoom,pan}=zoomAt(current.record,size,view,value,point));renderCanvas();
  }
  function changeZoom(factor, point={x:size.width/2,y:size.height/2}) {
    setZoom(zoom*factor,point);
  }
  function focusSelected(){if(!draft || loading || busy || navigating || gesture || polygonPoints.length || selected<0)return;({scale:zoom,pan}=focusView(current.record,size,draft.boxes[selected]));renderCanvas();canvas.focus();}
  function point(event) { const rect=canvas.getBoundingClientRect();return {x:event.clientX-rect.left,y:event.clientY-rect.top}; }
  function inside(p) { return p.x>=0 && p.y>=0 && p.x<=current.record.width && p.y<=current.record.height; }
  function errorWrap(action) {try {action();setError('');} catch (error) {setError(error.message);}render();}
  function finishPolygon(){if(!polygonPoints.length)return;errorWrap(()=>{draft.add(polygonBox(polygonPoints,polygonClass));polygonPoints=[];selected=draft.boxes.length-1;});}
  $('#polygon-finish').onclick=finishPolygon;$('#polygon-cancel').onclick=()=>{polygonPoints=[];render();};
  canvas.addEventListener('pointerdown',event=>{
    if (![0,1].includes(event.button) || !draft || loading || busy || navigating || gesture) return;
    event.preventDefault();canvas.focus();const screen=point(event), p=screenToImage(screen,view);pointer=screen;
    if (tool==='pan' || space || event.button===1) gesture={type:'pan',start:screen,pan:{...pan}};
    else if (tool==='draw') {
      if(closeReady()){finishPolygon();return;}
      if (!inside(p)) return;
      if (draft.state==='excluded') {setError('请先将排除状态恢复为未标注或已标注。');return;}
      if (!$('#draw-class').value) {setError('请先为当前任务添加类别并选择绘制类别。');return;}
      if(segmentation()){
        if(polygonPoints.length>=256){setError('每个多边形最多256个顶点，请先完成当前多边形。');return;}
        if(!polygonPoints.length)polygonClass=$('#draw-class').value;
        const last=polygonPoints.at(-1);if(!last||Math.hypot(last[0]-p.x,last[1]-p.y)*view.scale>2)polygonPoints.push([p.x,p.y]);render();return;
      }
      gesture={type:'draw',start:p,screen,classId:$('#draw-class').value};
    } else {
      let handle=null;
      if (selected>=0) for (const [name,hp] of Object.entries(segmentation()?Object.fromEntries(draft.boxes[selected].points.map(([x,y],i)=>[i,{x,y}])):handles(draft.boxes[selected]))) {
        const hpScreen=imageToScreen(hp,view);if (Math.hypot(hpScreen.x-screen.x,hpScreen.y-screen.y)<=9) {handle=name;break;}
      }
      if (handle===null) selected=segmentation()?hitPolygon(draft.boxes,p):hitBox(draft.boxes,p);
      if (selected>=0) gesture={type:handle!==null ? segmentation()?'vertex':'resize' : 'move',index:selected,handle,start:p,original:{...draft.boxes[selected]}};
    }
    if (gesture) canvas.setPointerCapture(event.pointerId);render();
  });
  canvas.addEventListener('pointermove',event=>{
    const screen=point(event);pointer=screen;
    if (!gesture) {if(polygonPoints.length)renderCanvas();return;}
    const p=screenToImage(screen,view);
    if (gesture.type==='pan') {pan={x:gesture.pan.x+screen.x-gesture.start.x,y:gesture.pan.y+screen.y-gesture.start.y};}
    else if (gesture.type==='draw') gesture.box=drawBox(gesture.start,p,current.record.width,current.record.height,gesture.classId);
    else if (gesture.type==='move') gesture.box=(segmentation()?movePolygon:moveBox)(gesture.original,p.x-gesture.start.x,p.y-gesture.start.y,current.record.width,current.record.height);
    else if (gesture.type==='vertex'){const points=gesture.original.points.map(p=>[...p]);points[Number(gesture.handle)]=[clamp(p.x,0,current.record.width),clamp(p.y,0,current.record.height)];gesture.box=polygonBox(points,gesture.original.class_id);}
    else gesture.box=resizeBox(gesture.original,gesture.handle,p,current.record.width,current.record.height);
    renderCanvas();
  });
  canvas.addEventListener('pointerup',event=>{
    if (!gesture) return;
    const end=gesture;gesture=null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    errorWrap(()=>{
      if (end.type==='draw' && end.box && end.box.width*view.scale>=3 && end.box.height*view.scale>=3) {draft.add(end.box);selected=draft.boxes.length-1;}
      else if (end.type!=='pan' && end.type!=='draw' && end.box) draft.replace(end.index,end.box);
    });
  });
  canvas.addEventListener('pointercancel',()=>{gesture=null;pointer=null;render();});
  canvas.addEventListener('pointerleave',()=>{pointer=null;if(!gesture)renderCanvas();});
  canvas.addEventListener('auxclick',event=>{if(event.button===1)event.preventDefault();});
  canvas.addEventListener('wheel',event=>{event.preventDefault();changeZoom(event.deltaY<0 ? 1.15 : 1/1.15,point(event));},{passive:false});
  for (const button of dialog.querySelectorAll('[data-tool]')) button.onclick=()=>{if (!gesture) setTool(button.dataset.tool);};
  $('#canvas-zoom-in').onclick=()=>changeZoom(1.25);$('#canvas-zoom-out').onclick=()=>changeZoom(0.8);$('#canvas-fit').onclick=resetView;
  $('#canvas-actual-size').onclick=()=>setZoom(1);$('#canvas-focus-selected').onclick=focusSelected;$('#canvas-zoom-preset').onchange=()=>{if($('#canvas-zoom-preset').value)setZoom(Number($('#canvas-zoom-preset').value));};
  const undo=()=>{if(polygonPoints.length){polygonPoints.pop();render();return;}if (!draft || loading || busy || navigating || gesture) return;draft.undo();selected=-1;setError('');render();};
  const redo=()=>{if (!draft || loading || busy || navigating || gesture) return;draft.redo();selected=-1;setError('');render();};
  const remove=()=>{if (draft && selected>=0 && !busy && !loading && !gesture) {draft.remove(selected);selected=-1;render();}};
  $('#canvas-undo').onclick=undo;$('#canvas-redo').onclick=redo;$('#box-delete').onclick=remove;
  $('#draw-class').onchange=()=>{if (current) drawingClass.remember(current,$('#draw-class').value);};
  $('#selected-class').onchange=()=>{if (draft && selected>=0) errorWrap(()=>{
    const id=$('#selected-class').value;
    draft.replace(selected,{...draft.boxes[selected],class_id:id});
    drawingClass.remember(current,id);$('#draw-class').value=id;
  });};
  $('#annotation-state').onchange=async()=>{
    if (!draft || loading || busy) return;
    if(polygonPoints.length){$('#annotation-state').value=draft.state;setError('请先完成或取消当前多边形。');return;}
    const state=$('#annotation-state').value;
    if (draft.boxes.length && ['unannotated','no_target'].includes(state)) {
      $('#annotation-state').value=draft.state;
      if (!await confirmClear('此操作会清除当前图片在本任务中的全部框。可撤销，点击保存后生效。')) return;
    }
    draft.setState(state);selected=-1;render();
  };
  async function save({ automatic=false } = {}) {
    if(polygonPoints.length){setError('请先完成或取消当前多边形，再保存。');return false;}
    if (!draft || loading || busy || gesture) return false;
    if (!draft.dirty) return true;
    const target=current, activeDraft=draft;busy=true;setError('');render();
    try {
      const data=await request(endpoint(target),{method:'PUT',body:{...activeDraft.snapshot(),revision:annotation.revision}});
      if (current!==target) return false;
      annotation=data.annotation;activeDraft.markSaved();changed=true;
      notify(automatic ? '当前图片的标注已自动保存' : '标注已保存，仅影响当前任务');return true;
    } catch (error) {if (current===target) setError(`${error.message} 当前框和编辑已保留。`);return false;}
    finally {busy=false;render();}
  }
  $('#annotation-save').onclick=save;
  function fillClasses() {
    const preferred=drawingClass.choose(current,classes);
    for (const id of ['draw-class','selected-class']) {
      const select=$(`#${id}`);select.replaceChildren();
      for (const record of classes) {const option=document.createElement('option');option.value=record.id;option.textContent=record.name;select.append(option);}
    }
    $('#draw-class').value=preferred;
  }
  async function load(target) {
    const version=++generation;pending=target;loading=true;setError('');$('#task-preview-status').textContent='正在读取图片与标注…';render();
    try {
      const source=`/api/projects/${target.projectId}/datasets/${target.task.dataset_id}/images/${target.record.id}`;
      const bitmap=new Image();bitmap.src=`${source}/view`;
      const [data,category]=await Promise.all([request(endpoint(target)),request(`/api/projects/${target.projectId}/tasks/${target.task.id}/classes`),bitmap.decode(),target.navigation.prepare()]);
      if (version!==generation || !dialog.open) return;
      if (bitmap.naturalWidth!==target.record.width || bitmap.naturalHeight!==target.record.height) throw new Error('图片尺寸与方向基准不一致，请检查源图片。');
      current=target;annotation=data.annotation;classes=category.classes;image=bitmap;
      draft=new (target.task.task_type==='segment'?PolygonDraft:BoxDraft)(annotation,target.record.width,target.record.height,classes.map(record=>record.id));polygonPoints=[];
      dialog.querySelector('[data-tool="draw"]').textContent=segmentation()?'绘制多边形 D':'绘制框 D';dialog.querySelector('.box-list-heading h3').textContent=segmentation()?'多边形':'矩形框';$('#box-delete').textContent=segmentation()?'删除选中多边形':'删除选中框';
      selected=-1;gesture=null;space=false;pointer=null;tool=classes.length ? 'draw' : 'select';fillClasses();
      $('#task-preview-title').textContent=target.record.filename;
      $('#task-preview-info').textContent=`${target.task.name} · ${target.record.width} × ${target.record.height} · ${target.task.dataset_name}`;
      $('#task-preview-download').href=`${source}/original`;$('#task-preview-status').textContent='';
      $('#canvas-retry').hidden=true;resetView();
    } catch (error) {
      if (version===generation) {setError(error.message || '图片读取失败，请重试。');$('#task-preview-status').textContent='读取失败，请重试。';$('#canvas-retry').hidden=false;}
    } finally {if (version===generation) {loading=false;resizeCanvas();render();}}
  }
  $('#canvas-retry').onclick=async()=>{if (!busy && pending && (!isDirty() || await beforeLeave())) load(pending);};
  async function navigate(delta) {
    if(polygonPoints.length){setError('请先完成或取消当前多边形，再切换图片。');return;}
    if (!current || busy || loading || navigating || gesture) return;
    const target=current;navigating=true;render();
    const advance=async()=>{
      if (current!==target || !dialog.open) return;
      loading=true;render();
      const record=await target.navigation.neighbour(delta,target.record.id);
      if (record && current===target && dialog.open) await load({...target,record});
    };
    try {
      if (delta>0) await saveBeforeNext({dirty:draft.dirty,save:()=>save({automatic:true}),advance});
      else if (await beforeLeave()) await advance();
    }
    catch (error) {setError(error.message);}
    finally {navigating=false;loading=false;render();}
  }
  $('#annotation-prev').onclick=()=>navigate(-1);$('#annotation-next').onclick=()=>navigate(1);
  dialog.addEventListener('keydown',event=>{
    if (!draft || loading || busy || navigating || $('#discard-dialog').open || event.target.closest('input,select,textarea,[contenteditable="true"]')) return;
    const key=event.key.toLowerCase();
    if(key===' '&&event.target===canvas&&!event.ctrlKey&&!event.metaKey&&!event.altKey){event.preventDefault();space=true;renderCanvas();return;}
    if(segmentation()&&polygonPoints.length){if(key==='escape'){event.preventDefault();event.stopPropagation();if(gesture?.type==='pan')gesture=null;else polygonPoints=[];render();return;}if(key==='enter'&&!gesture){event.preventDefault();finishPolygon();return;}if((key==='backspace'||key==='delete'||(event.ctrlKey&&key==='z'))&&!gesture){event.preventDefault();polygonPoints.pop();render();return;}}
    if ((event.ctrlKey || event.metaKey) && key==='s') {event.preventDefault();save();return;}
    if ((event.ctrlKey || event.metaKey) && key==='z') {event.preventDefault();event.shiftKey ? redo() : undo();return;}
    if ((event.ctrlKey || event.metaKey) && key==='y') {event.preventDefault();redo();return;}
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (key==='escape' && gesture) {event.preventDefault();event.stopPropagation();gesture=null;render();return;}
    if (key==='delete' || key==='backspace') {event.preventDefault();remove();}
    else if (['v','d','h'].includes(key)) {event.preventDefault();setTool({v:'select',d:'draw',h:'pan'}[key]);}
    else if (key==='pageup' || key==='pagedown') {event.preventDefault();navigate(key==='pageup' ? -1 : 1);}
    else if (['arrowleft','arrowright','arrowup','arrowdown'].includes(key) && selected>=0) {
      event.preventDefault();const step=event.shiftKey ? 10 : 1;
      errorWrap(()=>draft.replace(selected,(segmentation()?movePolygon:moveBox)(draft.boxes[selected],key==='arrowleft' ? -step : key==='arrowright' ? step : 0,key==='arrowup' ? -step : key==='arrowdown' ? step : 0,current.record.width,current.record.height)));
    } else if (key==='0') {event.preventDefault();resetView();}
    else if (key==='1') {event.preventDefault();setZoom(1);}
    else if (key==='f') {event.preventDefault();focusSelected();}
    else if (key==='+' || key==='=') {event.preventDefault();changeZoom(1.25);}
    else if (key==='-') {event.preventDefault();changeZoom(0.8);}
  });
  window.addEventListener('keyup',event=>{if (event.code==='Space') {space=false;if(dialog.open)renderCanvas();}});
  canvas.addEventListener('blur',()=>{space=false;if(dialog.open)renderCanvas();});
  window.addEventListener('blur',()=>{space=false;if(gesture?.type==='pan')gesture=null;if(dialog.open)render();});
  return {
    get dirty() {return isDirty();},get busy() {return busy;},get changed() {return changed;},
    open(target) { changed=false;current=null;pending=null;draft=null;annotation=null;image=null;classes=[];selected=-1;gesture=null;polygonPoints=[];load(target); },
    hide() {generation++;current=null;pending=null;draft=null;image=null;gesture=null;polygonPoints=[];space=false;loading=false;changed=false;},
  };
}
