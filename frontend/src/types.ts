export type Point = [number, number, number];
export type Arm = {joints_deg: Record<string, number>; targets_deg: Record<string, number>; points_m: Point[]; target_points_m: Point[]; gripper: string; tcp_target_m: Point | null};
export type Snapshot = {schema_version:1; sequence:number; status:'starting'|'running'|'error'; demo:boolean; token:string; age_ms:number; detected?:boolean; fps?:number; quality?:number; calibrated?:boolean; recording?:boolean; source?:string; resolution?:number[]; message?:string; skeleton_status?:string; mirrored?:boolean; arms?:Record<string,Arm>; body_m?:number[][]; gestures?:string[]; angles_deg?:Record<string,number|null>};
export type View = 'Perspective'|'Front'|'Top'|'Side';
export function validSnapshot(value: unknown): value is Snapshot {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return v.schema_version === 1 && typeof v.sequence === 'number' && typeof v.token === 'string' && typeof v.age_ms === 'number' && ['starting','running','error'].includes(String(v.status));
}
