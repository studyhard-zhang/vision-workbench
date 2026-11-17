export class ApiError extends Error {
  constructor(message, code, fields = {}) {
    super(message);
    this.code = code;
    this.fields = fields;
  }
}

export async function request(path, { method = 'GET', body, signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method, signal,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      cache: 'no-store',
    });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError('无法连接工作区。请检查本地服务是否仍在运行，当前输入已保留。', 'NETWORK_ERROR');
  }
  let data;
  try { data = await response.json(); }
  catch { throw new ApiError('工作区返回了无法识别的响应，请稍后重试。', 'INVALID_RESPONSE'); }
  if (!response.ok) {
    throw new ApiError(data.error?.message || '操作未完成，请重试。', data.error?.code, data.error?.fields);
  }
  return data;
}

export async function uploadTrainingResult(path,file) {
  let response;
  try{response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/zip'},body:file,cache:'no-store'});}
  catch{throw new ApiError('结果未确认接收，请保留本地ZIP并重试；同一结果不会重复登记。','NETWORK_ERROR');}
  let data;try{data=await response.json();}catch{throw new ApiError('结果响应无法识别，请保留ZIP并刷新记录。','INVALID_RESPONSE');}
  if(!response.ok)throw new ApiError(data.error?.message || '结果未接收，请重试。',data.error?.code);
  return data;
}
