// Task-scoped preferences; injected storage keeps private browsing fully usable.
export class DrawingClassPreference {
  constructor(storage = null) { this.storage=storage;this.values=new Map(); }
  key(target) { return `vision-workspace:draw-class:${target.projectId}:${target.task.id}`; }
  remember(target, id) {
    const key=this.key(target);this.values.set(key,id);
    try {this.storage?.setItem(key,id);} catch { /* Local storage is optional. */ }
  }
  choose(target, classes) {
    const key=this.key(target);let previous=this.values.get(key);
    if (!previous) {try {previous=this.storage?.getItem(key);} catch { /* Use the in-memory fallback. */ }}
    const id=classes.some(item=>item.id===previous) ? previous : classes[0]?.id || '';
    if (id) this.remember(target,id);
    return id;
  }
}

// A failed save must never advance the image or discard the draft.
export async function saveBeforeNext({ dirty, save, advance }) {
  if (dirty && !await save()) return false;
  await advance();return true;
}
