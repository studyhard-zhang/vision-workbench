import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DrawingClassPreference, saveBeforeNext } from '../web/editor-flow.mjs';
const a={projectId:'p',task:{id:'a'}},b={projectId:'p',task:{id:'b'}};
const classes=[{id:'first'},{id:'second'}];

test('chosen drawing class survives switching images, reopening and a new editor instance',()=>{
  const values=new Map(),storage={getItem:key=>values.get(key),setItem:(key,id)=>values.set(key,id)};
  const preference=new DrawingClassPreference(storage);
  assert.equal(preference.choose(a,classes),'first');preference.remember(a,'second');
  assert.equal(preference.choose(a,classes),'second');
  assert.equal(new DrawingClassPreference(storage).choose(a,classes),'second');
});
test('drawing preference is scoped to project and task and falls back after class deletion',()=>{
  const preference=new DrawingClassPreference();preference.remember(a,'second');
  assert.equal(preference.choose(b,classes),'first');
  assert.equal(preference.choose({...a,projectId:'other'},classes),'first');
  assert.equal(preference.choose(a,[{id:'first'}]),'first');
  assert.equal(preference.choose(a,[]),'');
});
test('drawing preference still works when local storage is unavailable',()=>{
  const preference=new DrawingClassPreference({getItem(){throw Error('blocked');},setItem(){throw Error('blocked');}});
  preference.remember(a,'second');assert.equal(preference.choose(a,classes),'second');
});
test('next image awaits a successful save before advancing',async()=>{
  const events=[];let finish;
  const result=saveBeforeNext({dirty:true,save:()=>{events.push('save');return new Promise(resolve=>{finish=resolve;});},advance:async()=>events.push('advance')});
  assert.deepEqual(events,['save']);finish(true);assert.equal(await result,true);assert.deepEqual(events,['save','advance']);
});
test('save failures and conflicts keep the current image and draft',async()=>{
  let advanced=false;
  assert.equal(await saveBeforeNext({dirty:true,save:async()=>false,advance:()=>{advanced=true;}}),false);
  assert.equal(advanced,false);
});
test('unchanged empty images advance without writing or becoming negative samples',async()=>{
  let saves=0,advances=0;
  assert.equal(await saveBeforeNext({dirty:false,save:()=>{saves++;},advance:()=>{advances++;}}),true);
  assert.equal(saves,0);assert.equal(advances,1);
});
