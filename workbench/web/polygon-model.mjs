import {BoxDraft,clamp} from './box-model.mjs';

export function polygonBox(points,class_id){
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
  return {class_id,x:Math.min(...xs),y:Math.min(...ys),width:Math.max(...xs)-Math.min(...xs),height:Math.max(...ys)-Math.min(...ys),points:points.map(p=>[...p])};
}
export function validPolygon(points,width,height){
  if(!Array.isArray(points)||points.length<3||points.length>256||points.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>typeof v!=='number'||!Number.isFinite(v))||p[0]<0||p[0]>width||p[1]<0||p[1]>height))return false;
  if(new Set(points.map(p=>JSON.stringify(p))).size!==points.length)return false;
  const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const on=(a,b,c)=>Math.abs(cross(a,b,c))<1e-9&&c[0]>=Math.min(a[0],b[0])&&c[0]<=Math.max(a[0],b[0])&&c[1]>=Math.min(a[1],b[1])&&c[1]<=Math.max(a[1],b[1]);
  for(let i=0;i<points.length;i++)for(let j=i+1;j<points.length;j++){
    if(j===i+1||i===0&&j===points.length-1)continue;
    const a=points[i],b=points[(i+1)%points.length],c=points[j],d=points[(j+1)%points.length];
    if(cross(a,b,c)*cross(a,b,d)<0&&cross(c,d,a)*cross(c,d,b)<0||on(a,b,c)||on(a,b,d)||on(c,d,a)||on(c,d,b))return false;
  }
  return Math.abs(points.reduce((s,a,i)=>s+a[0]*points[(i+1)%points.length][1]-points[(i+1)%points.length][0]*a[1],0))/2>=1;
}
export function movePolygon(shape,dx,dy,width,height){
  dx=clamp(dx,-shape.x,width-shape.x-shape.width);dy=clamp(dy,-shape.y,height-shape.y-shape.height);
  return polygonBox(shape.points.map(([x,y])=>[x+dx,y+dy]),shape.class_id);
}
export function hitPolygon(shapes,p){
  for(let k=shapes.length-1;k>=0;k--){const pts=shapes[k].points;let inside=false;
    for(let i=0,j=pts.length-1;i<pts.length;j=i++)if((pts[i][1]>p.y)!==(pts[j][1]>p.y)&&p.x<(pts[j][0]-pts[i][0])*(p.y-pts[i][1])/(pts[j][1]-pts[i][1])+pts[i][0])inside=!inside;
    if(inside)return k;
  }return -1;
}
export class PolygonDraft extends BoxDraft{
  valid(shape){return this.classIds.has(shape.class_id)&&validPolygon(shape.points,this.width,this.height);}
  add(shape){if(!this.valid(shape))throw Error('多边形至少3点、面积至少1像素，且不能越界、自交或重复顶点。');super.add(shape);}
  replace(index,shape){if(!this.valid(shape))throw Error('顶点修改无效：多边形不能越界、自交或退化。');super.replace(index,shape);}
}
