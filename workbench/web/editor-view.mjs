// View-only transforms. All annotations remain in canonical image coordinates.
export const MIN_SCALE=0.001,MAX_SCALE=64;
const bound=value=>Math.max(MIN_SCALE,Math.min(MAX_SCALE,value));
export function fitScale(image,size){return bound(Math.min(Math.max(1,size.width-30)/image.width,Math.max(1,size.height-30)/image.height));}
export function centeredView(image,size,scale,pan={x:0,y:0}){return {scale,x:(size.width-image.width*scale)/2+pan.x,y:(size.height-image.height*scale)/2+pan.y};}
export function zoomAt(image,size,view,next,point={x:size.width/2,y:size.height/2}){
  const scale=bound(next),ix=(point.x-view.x)/view.scale,iy=(point.y-view.y)/view.scale,center=centeredView(image,size,scale);
  return {scale,pan:{x:point.x-ix*scale-center.x,y:point.y-iy*scale-center.y}};
}
export function focusView(image,size,box){
  const scale=bound(Math.min(size.width*.65/box.width,size.height*.65/box.height));
  return {scale,pan:{x:(image.width/2-box.x-box.width/2)*scale,y:(image.height/2-box.y-box.height/2)*scale}};
}
