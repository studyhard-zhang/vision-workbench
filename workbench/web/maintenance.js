import {request} from './api.js';
const $=s=>document.querySelector(s);
const size=n=>`${(n/1024/1024).toFixed(2)} MiB`;
const labels={assets:'原图与预览',versions:'冻结版本',training:'训练产物',models:'正式模型',evaluations:'独立评估',inferences:'图片推理', 'inference-inputs':'推理输入',backups:'数据库迁移备份','project-backups':'项目备份',database:'业务数据库（含WAL）',releases:'程序发布包'};
let busy=false;
function message(text,error=false){$('#maintenance-message').textContent=text;$('#maintenance-message').className=error?'notice error-notice':'notice';}
function setBusy(value){busy=value;for(const s of ['#maintenance-backup','#maintenance-restore','#maintenance-cleanup','#maintenance-close'])$(s).disabled=value;}
async function refresh(){
 const [{usage,backups,restores},{projects}]=await Promise.all([request('/api/maintenance'),request('/api/projects')]);
 $('#maintenance-space').textContent=`数据目录：${usage.data_directory}\n可用空间：${size(usage.free_bytes)}；安全预留：${size(usage.reserve_bytes)}。恢复副本保存在独立目录：${usage.restore_root}`;
 const table=$('#maintenance-usage');table.replaceChildren();
 for(const [key,value] of Object.entries(usage.groups)){const row=document.createElement('tr');for(const text of [labels[key]||key,String(value.files),size(value.bytes)]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}table.append(row);}
 if(usage.groups.assets.detail){const d=usage.groups.assets.detail;$('#maintenance-space').textContent+=`\n素材分项：原图 ${size(d['original.'])} / 全尺寸预览 ${size(d['display.'])} / 缩略图 ${size(d['thumbnail.'])}`;}
 const select=$('#maintenance-project');const prior=select.value;select.replaceChildren();for(const p of projects){const o=document.createElement('option');o.value=p.id;o.textContent=p.name;select.append(o);}if(projects.some(p=>p.id===prior))select.value=prior;
 $('#maintenance-cleanup').textContent=`清理过期临时文件（${usage.cleanup_candidates}个）`;
 const list=$('#maintenance-backups');list.replaceChildren();for(const b of backups){const p=document.createElement('p'),a=document.createElement('a');a.className='button secondary';a.href=`/api/maintenance/backups/${b.id}`;a.download='';a.textContent=`下载 ${b.project_name} · ${size(b.bytes)} · ${new Date(b.created_at).toLocaleString('zh-CN')}`;p.append(a);list.append(p);}
 const restored=$('#maintenance-restores');restored.replaceChildren();for(const r of restores){const p=document.createElement('p');p.textContent=`已恢复 · ${new Date(r.restored_at).toLocaleString('zh-CN')}\n${r.data_directory}`;restored.append(p);}
 $('#maintenance-events').textContent=usage.events.map(e=>`${e.time} · ${e.kind}${e.released_bytes!==undefined?' · 释放 '+size(e.released_bytes):''}`).join('\n')||'暂无维护记录。';
}
$('#open-maintenance').onclick=async()=>{$('#maintenance-dialog').showModal();message('正在读取空间与备份记录…');try{await refresh();message('备份前请保存标注并关闭其他编辑窗口；等待和活动计算任务会阻止备份。');}catch(e){message(e.message,true);}};
$('#maintenance-close').onclick=()=>{if(!busy)$('#maintenance-dialog').close();};$('#maintenance-dialog').addEventListener('cancel',e=>{if(busy)e.preventDefault();});
$('#maintenance-backup').onclick=async()=>{if(busy)return;const id=$('#maintenance-project').value;if(!id)return;setBusy(true);message('正在创建一致性备份；本工作区暂时禁止写入，请保持页面打开。');try{const {backup}=await request(`/api/projects/${id}/backup`,{method:'POST',body:{}});await refresh();message(`备份完成：${backup.files}个文件，${size(backup.bytes)}。SHA-256：${backup.sha256}。请下载到外部存储；同盘副本不等同外部备份。`);}catch(e){message(e.message,true);}finally{setBusy(false);}};
$('#maintenance-cleanup').onclick=async()=>{if(busy)return;setBusy(true);try{const r=await request('/api/maintenance/cleanup',{method:'POST',body:{confirmation:'清理过期临时文件'}});await refresh();message(`清理${r.files}个过期暂存文件，释放${size(r.released_bytes)}；原图、版本、日志和权重保留。`);}catch(e){message(e.message,true);}finally{setBusy(false);}};
$('#maintenance-restore').onclick=async()=>{if(busy)return;const file=$('#maintenance-file').files[0];if(!file){message('请选择本软件生成的项目备份ZIP。',true);return;}if(file.size>2*1024*1024*1024){message('备份ZIP不能超过2GiB。',true);return;}setBusy(true);message('正在上传、校验并恢复到新目录；当前项目不会被覆盖。');try{const response=await fetch('/api/maintenance/restore',{method:'POST',headers:{'Content-Type':'application/zip'},body:file});const data=await response.json();if(!response.ok)throw Error(data.error?.message||'恢复未完成');const r=data.restore;await refresh();message(`恢复成功：${r.project_name}\n数据目录：${r.data_directory}\n原始ID与历史SHA保留；没有恢复SSH配置或启动计算。使用同版本程序和空闲端口，通过 --data-dir 指向该目录即可检查副本。`);}catch(e){message(e.message,true);}finally{setBusy(false);}};
