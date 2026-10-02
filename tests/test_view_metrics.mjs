import test from 'node:test';
import assert from 'node:assert/strict';
import {errorMetrics, selectVideos, relatedBlocks, fetchAllPages} from '../serving_app/static/view-metrics.js';
const row=(p,a)=>({predicted_views_day7:p,target_views_day7:a});
test('missing actual is pending; actual zero is not missing',()=>{
  assert.equal(errorMetrics(row(12,null)),null);
  assert.equal(errorMetrics(row(0,0)).ratio,1);
  assert.equal(errorMetrics(row(100,0)).review,true);
});
test('related drift requires block id AND mode AND model version',()=>{
  const videos=[{block_id:2,model_version:'1',mode:'live'}];
  const blocks=[{block_id:2,model_version:'1',mode:'live'},{block_id:2,model_version:'2',mode:'live'},{block_id:2,model_version:'1',mode:'simulation'}];
  assert.deepEqual(relatedBlocks(videos,blocks),[blocks[0]]);
  assert.equal(relatedBlocks([{block_id:null}],blocks).length,0);
});
test('pending records beyond first server page are included',async()=>{
  const result=await fetchAllPages(async({page})=>({total:101,items:page===1?Array.from({length:100},()=>row(10,10)):[row(10,null)]}));
  assert.equal(result.items.length,101);
  assert.equal(selectVideos(result.items,'pending','recent').length,1);
});
test('exact log ratio boundaries and signed errors',()=>{
  assert.equal(errorMetrics(row(201,100)).review,true);
  assert.equal(errorMetrics(row(100,201)).review,true);
  assert.equal(errorMetrics(row(200,100)).review,false);
  assert.equal(errorMetrics(row(100,201)).delta,-101);
});
test('local filtering and sorting preserve original API page',()=>{
  const rows=[row(10,null),row(110,100),row(400,100),row(1000,600)];
  assert.equal(selectVideos(rows,'review','recent').length,1);
  assert.equal(selectVideos(rows,'pending','recent').length,1);
  assert.equal(selectVideos(rows,'all','relative')[0],rows[2]);
  assert.equal(selectVideos(rows,'all','absolute')[0],rows[3]);
  assert.equal(rows[0].target_views_day7,null);
});
