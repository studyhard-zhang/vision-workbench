// Missing measurements remain gaps; even a single epoch gets a visible point.
export function chartGeometry(rows, keys, bounded=false){
  const left=56,right=620,top=18,bottom=212;
  const first=rows[0]?.epoch ?? 1,last=rows.at(-1)?.epoch ?? first;
  const values=rows.flatMap(row=>keys.map(key=>row[key])).filter(value=>typeof value==='number' && Number.isFinite(value));
  const max=bounded?1:Math.max(...values,0)*1.08 || 1;
  const x=epoch=>first===last?(left+right)/2:left+(epoch-first)/(last-first)*(right-left);
  const y=value=>bottom-value/max*(bottom-top);
  return {left,right,top,bottom,max,first,last,
    ticks:[0,.25,.5,.75,1].map(fraction=>({value:max*fraction,y:y(max*fraction)})),
    series:keys.map(key=>{
      const segments=[];let current=[],previous=null;
      for(const row of rows){
        const value=row[key];
        if(typeof value!=='number' || !Number.isFinite(value) || (previous!==null && row.epoch!==previous+1)){
          if(current.length)segments.push(current);current=[];
        }
        if(typeof value==='number' && Number.isFinite(value))current.push({epoch:row.epoch,value,x:x(row.epoch),y:y(value)});
        previous=row.epoch;
      }
      if(current.length)segments.push(current);
      return {key,segments};
    })};
}

export function formatMetric(value,compact=false){
  if(typeof value!=='number' || !Number.isFinite(value))return '未记录';
  if(compact){
    if(value!==0 && Math.abs(value)<.0001)return value.toExponential(2);
    return String(Number(value.toPrecision(4)));
  }
  return String(value);
}
