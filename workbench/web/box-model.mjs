// Geometry always uses EXIF-normalized image pixels; zoom and pan are view-only.
export const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
const copy = value => JSON.parse(JSON.stringify(value));
const key = value => JSON.stringify({ state: value.state, boxes: value.boxes });
const round = n => Math.round(n * 1000) / 1000;

export function screenToImage(point, view) {
  return { x: (point.x - view.x) / view.scale, y: (point.y - view.y) / view.scale };
}
export function imageToScreen(point, view) {
  return { x: point.x * view.scale + view.x, y: point.y * view.scale + view.y };
}
export function drawBox(a, b, width, height, classId) {
  const x1 = round(clamp(Math.min(a.x,b.x),0,width)), y1 = round(clamp(Math.min(a.y,b.y),0,height));
  const x2 = round(clamp(Math.max(a.x,b.x),0,width)), y2 = round(clamp(Math.max(a.y,b.y),0,height));
  return { class_id: classId, x: x1, y: y1, width: round(x2-x1), height: round(y2-y1) };
}
export function moveBox(box, dx, dy, width, height) {
  return { ...box, x: round(clamp(box.x+dx,0,width-box.width)), y: round(clamp(box.y+dy,0,height-box.height)) };
}
export function resizeBox(box, handle, point, width, height) {
  let x1=box.x, y1=box.y, x2=box.x+box.width, y2=box.y+box.height;
  if (handle.includes('w')) x1=clamp(point.x,0,x2-1);
  if (handle.includes('e')) x2=clamp(point.x,x1+1,width);
  if (handle.includes('n')) y1=clamp(point.y,0,y2-1);
  if (handle.includes('s')) y2=clamp(point.y,y1+1,height);
  return drawBox({x:x1,y:y1},{x:x2,y:y2},width,height,box.class_id);
}
export function handles(box) {
  const x=box.x, y=box.y, r=x+box.width, b=y+box.height, cx=(x+r)/2, cy=(y+b)/2;
  return { nw:{x,y}, n:{x:cx,y}, ne:{x:r,y}, e:{x:r,y:cy}, se:{x:r,y:b}, s:{x:cx,y:b}, sw:{x,y:b}, w:{x,y:cy} };
}
export function hitBox(boxes, point) {
  for (let i=boxes.length-1;i>=0;i--) {
    const b=boxes[i]; if (point.x>=b.x && point.x<=b.x+b.width && point.y>=b.y && point.y<=b.y+b.height) return i;
  }
  return -1;
}

export class BoxDraft {
  constructor(annotation, width, height, classIds) {
    this.width=width; this.height=height; this.classIds=new Set(classIds);
    this.state=annotation.state; this.boxes=copy(annotation.boxes);
    this.baseline=key(this); this.past=[]; this.future=[];
  }
  get dirty() { return key(this)!==this.baseline; }
  snapshot() { return copy({ state:this.state, boxes:this.boxes }); }
  valid(box) {
    return this.classIds.has(box.class_id) && ['x','y','width','height'].every(k=>typeof box[k]==='number' && Number.isFinite(box[k])) &&
      box.x>=0 && box.y>=0 && box.width>=1 && box.height>=1 && box.x+box.width<=this.width+1e-7 && box.y+box.height<=this.height+1e-7;
  }
  commit(next) {
    if (key(this)===key(next)) return false;
    this.past.push(this.snapshot()); if (this.past.length>100) this.past.shift();
    this.future=[]; this.restore(next); return true;
  }
  restore(value) { this.state=value.state; this.boxes=copy(value.boxes); }
  add(box) {
    if (this.boxes.length>=200) throw new Error('每张图片最多 200 个框。');
    if (!this.valid(box)) throw new Error('矩形框至少为 1 × 1 像素，且须位于图片范围内。');
    this.commit({state:'annotated',boxes:[...this.boxes,box]});
  }
  replace(index, box) {
    if (!this.boxes[index] || !this.valid(box)) throw new Error('矩形框坐标或类别无效。');
    const boxes=copy(this.boxes); boxes[index]=box; this.commit({state:this.state,boxes});
  }
  remove(index) {
    if (!this.boxes[index]) return;
    const boxes=this.boxes.filter((_,i)=>i!==index);
    this.commit({state:!boxes.length && this.state==='annotated' ? 'unannotated' : this.state,boxes});
  }
  setState(state) {
    if (!['unannotated','annotated','no_target','excluded'].includes(state)) throw new Error('图片状态无效。');
    if (state==='annotated' && !this.boxes.length) throw new Error('已标注须包含至少一个框。');
    this.commit({state,boxes:['unannotated','no_target'].includes(state) ? [] : this.boxes});
  }
  undo() { if (this.past.length) { this.future.push(this.snapshot()); this.restore(this.past.pop()); } }
  redo() { if (this.future.length) { this.past.push(this.snapshot()); this.restore(this.future.pop()); } }
  markSaved() { this.baseline=key(this); }
}
