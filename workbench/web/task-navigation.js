const sections=['images','preparation','training','inference'];

export function createTaskNavigation(){
  const buttons=[...document.querySelectorAll('[data-task-section]')].filter(b=>sections.includes(b.dataset.taskSection));
  document.querySelector('#task-panel-preparation').append(document.querySelector('#task-panel-versions'));
  const panels=sections.map(id=>document.querySelector(`#task-panel-${id}`));
  let key=null,selected='images';
  const memory=new Map();
  function select(id,{focus=false,remember=true}={}){
    if(id==='versions')id='preparation';if(id==='models')id='training';
    if(!sections.includes(id))id='images';
    selected=id;
    for(const button of buttons){const active=button.dataset.taskSection===id;button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;}
    for(const panel of panels)panel.hidden=panel.id!==`task-panel-${id}`;
    document.querySelector('#task-panel-versions').hidden=id!=='preparation';
    document.querySelector('#task-panel-models').hidden=true;
    if(key&&remember){memory.set(key,id);try{sessionStorage.setItem(key,id);}catch{}}
    const button=buttons.find(b=>b.dataset.taskSection===id);
    if(focus){button.focus();button.scrollIntoView({block:'nearest',inline:'nearest'});}
    if(id==='inference')requestAnimationFrame(()=>{if(selected==='inference'&&button.getClientRects().length)button.closest('nav').scrollIntoView({block:'start'});});
  }
  for(const button of buttons){
    button.addEventListener('click',()=>select(button.dataset.taskSection));
    button.addEventListener('keydown',event=>{
      const index=sections.indexOf(button.dataset.taskSection);let next;
      if(event.key==='ArrowRight')next=(index+1)%sections.length;
      if(event.key==='ArrowLeft')next=(index+sections.length-1)%sections.length;
      if(event.key==='Home')next=0;
      if(event.key==='End')next=sections.length-1;
      if(next===undefined)return;event.preventDefault();select(sections[next],{focus:true});
    });
  }
  document.addEventListener('task-workspace-select',event=>{if(key&&sections.includes(event.detail))select(event.detail);});
  return {
    open(projectId,taskId,taskType='detect'){
      buttons.find(b=>b.dataset.taskSection==='inference').hidden=false;
      key=`vision-task-section:${projectId}:${taskId}`;
      let stored=memory.get(key);try{stored=sessionStorage.getItem(key)||stored;}catch{}
      select(stored||'images',{remember:false});
    },
    select,
    get selected(){return selected;}
  };
}
