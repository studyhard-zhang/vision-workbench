import test from 'node:test';
import assert from 'node:assert/strict';
import {PolygonDraft,polygonBox,validPolygon,movePolygon,hitPolygon} from '../web/polygon-model.mjs';
test('simple concave polygons remain valid while self intersections, duplicates and bounds are rejected',()=>{
  assert(validPolygon([[0,0],[100,0],[40,40],[100,100],[0,100]],100,100));
  for(const points of [[[0,0],[100,100],[0,100],[100,0]],[[0,0],[100,0],[0,0]],[[0,0],[101,0],[0,100]],[[0,0],[20,0],[40,0]]])assert(!validPolygon(points,100,100));
});
test('polygon moves clamp all vertices, select by actual contour and preserve undo and saved baseline',()=>{
  const draft=new PolygonDraft({state:'unannotated',boxes:[]},100,100,['c']);const shape=polygonBox([[10,10],[80,10],[10,80]],'c');draft.add(shape);draft.markSaved();
  assert.equal(hitPolygon([shape],{x:70,y:70}),-1);assert.equal(hitPolygon([shape],{x:20,y:20}),0);
  draft.replace(0,movePolygon(shape,500,-500,100,100));assert.deepEqual(draft.boxes[0].points,[[30,0],[100,0],[30,70]]);assert(draft.dirty);
  draft.undo();assert(!draft.dirty);draft.redo();assert(draft.dirty);draft.setState('no_target');assert.equal(draft.boxes.length,0);draft.undo();assert.equal(draft.boxes.length,1);
});
test('invalid vertex edits preserve the previous draft and a separate image has no shared points',()=>{
  const draft=new PolygonDraft({state:'unannotated',boxes:[]},100,100,['c']);draft.add(polygonBox([[0,0],[100,0],[100,100],[0,100]],'c'));const before=draft.snapshot();
  assert.throws(()=>draft.replace(0,polygonBox([[0,0],[100,100],[100,0],[0,100]],'c')));assert.deepEqual(draft.snapshot(),before);
  const copy=new PolygonDraft(before,100,100,['c']);copy.boxes[0].points[0][0]=20;assert.equal(draft.boxes[0].points[0][0],0);
});
