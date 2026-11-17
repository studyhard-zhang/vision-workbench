import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BoxDraft, drawBox, moveBox, resizeBox, screenToImage, imageToScreen, hitBox } from '../web/box-model.mjs';
const empty={state:'unannotated',boxes:[]};
const box={class_id:'a',x:20,y:10,width:50,height:30};
const draft=()=>new BoxDraft(empty,200,100,['a','b']);

test('screen/image round trip preserves canonical coordinates under zoom and pan',()=>{
  for (const scale of [0.125,0.67,1,2.8,8]) {
    const view={scale,x:-73.25,y:189.5}, p={x:153.777,y:83.3};
    const result=screenToImage(imageToScreen(p,view),view);
    assert.ok(Math.abs(result.x-p.x)<1e-10);assert.ok(Math.abs(result.y-p.y)<1e-10);
  }
});
test('drawing in reverse and beyond edges clamps both endpoints',()=>{
  assert.deepEqual(drawBox({x:220,y:120},{x:-10,y:20},200,100,'a'),{class_id:'a',x:0,y:20,width:200,height:80});
});
test('moving preserves box size at every boundary',()=>{
  assert.deepEqual(moveBox(box,300,300,200,100),{...box,x:150,y:70});
  assert.deepEqual(moveBox(box,-300,-300,200,100),{...box,x:0,y:0});
});
test('eight resize handles keep opposite sides fixed and cannot invert',()=>{
  for (const handle of ['nw','n','ne','e','se','s','sw','w']) {
    const result=resizeBox(box,handle,{x:1000,y:-1000},200,100);
    assert.ok(result.x>=0 && result.y>=0 && result.x+result.width<=200 && result.y+result.height<=100);
    assert.ok(result.width>=1 && result.height>=1);
    if (!handle.includes('w')) assert.equal(result.x,box.x);
    if (!handle.includes('n')) assert.equal(result.y,box.y);
  }
});
test('selection chooses last drawn overlapping box',()=>{
  assert.equal(hitBox([box,{...box,class_id:'b'}],{x:40,y:20}),1);
  assert.equal(hitBox([box],{x:0,y:0}),-1);
});
test('draw, reclassify, move and delete support undo and redo',()=>{
  const value=draft();value.add(box);value.replace(0,{...box,class_id:'b'});value.replace(0,{...box,class_id:'b',x:40});
  assert.equal(value.state,'annotated');value.remove(0);assert.equal(value.state,'unannotated');
  value.undo();assert.equal(value.boxes[0].x,40);value.undo();assert.equal(value.boxes[0].class_id,'b');value.undo();assert.equal(value.boxes[0].class_id,'a');
  value.redo();assert.equal(value.boxes[0].class_id,'b');
});
test('last box deletion differs from explicit no-target and state change is undoable',()=>{
  const value=draft();value.add(box);value.setState('no_target');assert.equal(value.boxes.length,0);
  value.undo();assert.equal(value.state,'annotated');assert.equal(value.boxes.length,1);
  value.remove(0);assert.equal(value.state,'unannotated');
});
test('excluding retains boxes and resuming restores valid annotation',()=>{
  const value=draft();value.add(box);value.setState('excluded');assert.deepEqual(value.boxes,[box]);
  value.setState('annotated');assert.deepEqual(value.boxes,[box]);
});
test('saved baseline and per-image isolation do not mutate loaded annotation',()=>{
  const loaded={state:'annotated',boxes:[box]},first=new BoxDraft(loaded,200,100,['a','b']),second=new BoxDraft(loaded,200,100,['a','b']);
  first.replace(0,{...box,x:30});assert.equal(first.dirty,true);assert.equal(second.dirty,false);assert.equal(loaded.boxes[0].x,20);
  first.markSaved();assert.equal(first.dirty,false);first.undo();assert.equal(first.dirty,true);
});
test('invalid geometry, unsupported state, missing classes and 201st box are rejected',()=>{
  const value=draft();assert.throws(()=>value.setState('annotated'));assert.throws(()=>value.setState('done'));
  for (const invalid of [{...box,width:0},{...box,x:NaN},{...box,x:-1},{...box,class_id:'other'},{...box,width:500}]) assert.throws(()=>value.add(invalid));
  for (let i=0;i<200;i++) value.add(box);assert.throws(()=>value.add(box));assert.equal(value.boxes.length,200);
});
test('new edits discard redo branch and undo history remains bounded',()=>{
  const value=draft();value.add(box);value.undo();value.add({...box,class_id:'b'});assert.equal(value.future.length,0);
  for (let i=0;i<150;i++) value.replace(0,{...value.boxes[0],x:20+i%2});assert.equal(value.past.length,100);
});
