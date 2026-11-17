import {test} from 'node:test';import assert from 'node:assert/strict';
import {fitScale,centeredView,zoomAt,focusView,MAX_SCALE} from '../web/editor-view.mjs';
import {imageToScreen,screenToImage,drawBox} from '../web/box-model.mjs';
const image={width:4000,height:3000},size={width:1000,height:800};
test('one-to-one is actual source pixels while fit can be below one',()=>{
  assert(fitScale(image,size)<1);const view=centeredView(image,size,1);assert.equal(imageToScreen({x:101,y:0},view).x-imageToScreen({x:100,y:0},view).x,1);
});
test('zoom at the pointer preserves the same canonical image coordinate up to 6400 percent',()=>{
  let view=centeredView(image,size,fitScale(image,size));const mouse={x:233,y:271},before=screenToImage(mouse,view);
  for(const next of [1,8,32,64,1000,.01]){const change=zoomAt(image,size,view,next,mouse);view=centeredView(image,size,change.scale,change.pan);const after=screenToImage(mouse,view);assert(Math.abs(after.x-before.x)<1e-8);assert(Math.abs(after.y-before.y)<1e-8);assert(view.scale<=64);}
});
test('focusing a two-pixel defect centers it and reaches the source pixel zoom limit',()=>{
  const box={x:1234,y:2222,width:2,height:2};const change=focusView(image,size,box),view=centeredView(image,size,change.scale,change.pan);assert.equal(change.scale,MAX_SCALE);assert.deepEqual(imageToScreen({x:1235,y:2223},view),{x:500,y:400});
  const a=screenToImage({x:436,y:336},view),b=screenToImage({x:564,y:464},view);assert.deepEqual(drawBox(a,b,image.width,image.height,'defect'),{class_id:'defect',...box});
});
