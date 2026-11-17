import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chartGeometry,formatMetric } from '../web/training-chart.mjs';

test('missing observations and missing epochs break lines rather than drawing fabricated continuity',()=>{
  const {series}=chartGeometry([{epoch:1,v:1},{epoch:2,v:null},{epoch:3,v:3},{epoch:5,v:5}],['v']);
  assert.deepEqual(series[0].segments.map(segment=>segment.map(p=>p.epoch)),[[1],[3],[5]]);
});
test('a single epoch including a genuine zero still gets finite visible coordinates',()=>{
  const g=chartGeometry([{epoch:1,v:0}],['v']);const point=g.series[0].segments[0][0];
  assert.equal(point.x,(g.left+g.right)/2);assert.equal(point.y,g.bottom);assert.ok(g.max>0);
});
test('metrics share the zero-to-one scale and plot exact source numbers',()=>{
  const g=chartGeometry([{epoch:1,p:0,r:1},{epoch:50,p:.72725,r:.99}],['p','r'],true);
  assert.equal(g.max,1);assert.equal(g.series[0].segments[0][0].y,g.bottom);
  assert.equal(g.series[1].segments[0][0].y,g.top);
  assert.equal(g.series[0].segments[1][0].value,.72725);
});
test('learning-rate precision and unavailable values remain distinct in the raw table',()=>{
  assert.equal(formatMetric(.000729904), '0.000729904');assert.equal(formatMetric(0),'0');
  assert.equal(formatMetric(null),'未记录');assert.equal(formatMetric(NaN),'未记录');
});
