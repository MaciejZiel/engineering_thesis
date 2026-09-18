export type Point = [number, number, number];
export type Arm = {joints_deg: Record<string, number>; targets_deg: Record<string, number>; points_m: Point[]; target_points_m: Point[]; gripper: string; tcp_target_m: Point | null};
export type Snapshot = {schema_version:1; sequence:number; status:'starting'|'running'|'error'; demo:boolean; token:string; age_ms:number; detected?:boolean; fps?:number; quality?:number; calibrated?:boolean; recording?:boolean; source?:string; resolution?:number[]; message?:string; skeleton_status?:string; mirrored?:boolean; arms?:Record<string,Arm>; body_m?:number[][]; gestures?:string[]; angles_deg?:Record<string,number|null>};
export type View = 'Perspective'|'Front'|'Top'|'Side';
export function validSnapshot(value: unknown): value is Snapshot {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  const finite = (n:unknown):n is number => typeof n === 'number' && Number.isFinite(n);
  const record = (r:unknown):r is Record<string,unknown> => !!r && typeof r === 'object' && !Array.isArray(r);
  const points = (p:unknown) => Array.isArray(p) && p.every(point=>Array.isArray(point) && point.length >= 3 && point.every(finite));
  if (v.schema_version !== 1 || !finite(v.sequence) || typeof v.token !== 'string' || !finite(v.age_ms) || !['starting','running','error'].includes(String(v.status))) return false;
  if (v.arms !== undefined && (!record(v.arms) || !Object.values(v.arms).every(arm=>record(arm) && record(arm.joints_deg) && Object.values(arm.joints_deg).every(finite) && record(arm.targets_deg) && Object.values(arm.targets_deg).every(finite) && points(arm.points_m) && points(arm.target_points_m)))) return false;
  if (v.body_m !== undefined && !points(v.body_m)) return false;
  if (v.gestures !== undefined && (!Array.isArray(v.gestures) || !v.gestures.every(g=>typeof g==='string'))) return false;
  return ['fps','quality'].every(key=>v[key]===undefined || finite(v[key]));
}
