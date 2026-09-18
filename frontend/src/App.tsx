import {lazy, Suspense, useEffect, useRef, useState, type ReactNode} from 'react';
import {Activity, ArrowDownToLine, ArrowUpRight, Box, Camera, Check, ChevronRight, Circle, CircleHelp, Crosshair, Expand, Focus, Gauge, Grid2X2, Layers, ListFilter, Monitor, Move3D, Radio, RotateCcw, ScanLine, Settings2, SlidersHorizontal, Square, Unplug, Video, X} from 'lucide-react';
import {useTelemetry} from './useTelemetry';
import type {Snapshot, View} from './types';
const Scene=lazy(()=>import('./Scene'));
const tabs=[['workspace','Workspace',Box],['calibration','Calibration',ScanLine],['sessions','Session',Video],['diagnostics','Diagnostics',Activity]] as const;
type Tab=typeof tabs[number][0];
type Preferences={grid:boolean;targets:boolean;body:boolean};
const defaultPrefs:Preferences={grid:true,targets:false,body:true};
function loadPrefs():Preferences{try{return {...defaultPrefs,...JSON.parse(localStorage.getItem('motion-twin-view') || '{}')};}catch{return defaultPrefs;}}

export default function App(){
  const {state,connected,error}=useTelemetry();
  const [tab,setTab]=useState<Tab>('workspace');
  const [view,setView]=useState<View>('Perspective');
  const [prefs,setPrefs]=useState(loadPrefs);
  const [reset,setReset]=useState(0);
  const [modal,setModal]=useState<'settings'|'help'|'calibrate'|null>(null);
  const [notice,setNotice]=useState('');
  const [pending,setPending]=useState(false);
  const [focus,setFocus]=useState(false);
  const ready=connected&&state?.status==='running';
  const canAct=ready&&!state?.demo;
  const quality=Math.round((state?.quality??0)*100);
  const recording=!!state?.recording;
  useEffect(()=>{try{localStorage.setItem('motion-twin-view',JSON.stringify(prefs));}catch{/* Local preferences are optional. */}},[prefs]);
  useEffect(()=>{if(!notice)return;const id=setTimeout(()=>setNotice(''),6500);return()=>clearTimeout(id);},[notice]);
  async function command(name:string){
    if(!canAct||pending)return;setPending(true);
    try{const response=await fetch(`/api/v1/commands/${name}`,{method:'POST',headers:{'x-session-token':state!.token}});const result=await response.json();if(!response.ok)throw new Error(result.detail||'Action could not be completed');setNotice(result.message);}
    catch(e){setNotice(e instanceof Error?e.message:'Unable to reach the tracking engine');}
    finally{setPending(false);}
  }
  useEffect(()=>{
    const key=(event:KeyboardEvent)=>{if(event.ctrlKey||event.metaKey||event.altKey||event.repeat||modal||/INPUT|SELECT|TEXTAREA/.test((event.target as HTMLElement).tagName))return;
      if(event.key==='c'){event.preventDefault();setModal('calibrate');}
      if(event.key==='r'){event.preventDefault();void command('record');}
      if(event.key==='v'){event.preventDefault();setFocus(value=>!value);}
      if(event.key==='?'){event.preventDefault();setModal('help');}
    };window.addEventListener('keydown',key);return()=>window.removeEventListener('keydown',key);
  });
  const mode=state?.demo?'Demonstration':'Simulation';
  const tracking=state?.detected?'Person detected':'Waiting for person';
  const toggle=(key:keyof Preferences)=>setPrefs(value=>({...value,[key]:!value[key]}));
  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#workspace" onClick={()=>setTab('workspace')} aria-label="Motion Twin workspace"><span className="brand-mark"><Move3D size={21}/></span><span>Motion Twin<small>DUAL ARM WORKSPACE</small></span></a>
      <div className="project-switch"><span className="project-icon"><Box size={17}/></span><div>UR7e laboratory<small>Local workspace</small></div><ChevronRight size={14}/></div>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">{tabs.map(([id,label,Icon])=><button key={id} className={`nav-item ${tab===id?'selected':''}`} aria-current={tab===id?'page':undefined} onClick={()=>{setTab(id);setFocus(false);}}><Icon size={17}/><span>{label}</span>{id==='sessions'&&recording&&<i className="dot red"/>}</button>)}</nav>
      <div className="sidebar-spacer"/>
      <div className="connection-summary"><span className={`dot ${connected?'green':''}`}/><div>{connected?'Engine connected':'Engine offline'}<small>Python · local connection</small></div></div>
      <button className="nav-item" onClick={()=>setModal('settings')}><Settings2 size={17}/><span>Preferences</span></button>
      <button className="nav-item" onClick={()=>setModal('help')}><CircleHelp size={17}/><span>Keyboard shortcuts</span><kbd>?</kbd></button>
      <div className="sidebar-bottom"><span className="avatar">MT</span><div>Operator workspace<small>Development preview</small></div></div>
    </aside>
    <div className="application">
      <header className="topbar"><div className="breadcrumb">Laboratory<ChevronRight size={13}/><span>{tabs.find(t=>t[0]===tab)?.[1]}</span></div><div className="topbar-right"><span className="mode-label"><span className="dot amber"/>{mode}</span><span className="local-label"><Monitor size={13}/> Local</span><button className="icon-button" title="Enter fullscreen" aria-label="Enter fullscreen" onClick={()=>{const result=document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();void result.catch(()=>setNotice('Fullscreen is unavailable in this browser.'));}}><Expand size={16}/></button></div></header>
      <main id="main-content">
        <div className="page-heading"><div><div className="heading-line"><h1>{tab==='workspace'?'Motion workspace':tab==='calibration'?'Operator calibration':tab==='sessions'?'Session recording':'System diagnostics'}</h1><span className="version-tag">2 × UR7e</span></div><p>{tab==='workspace'?'Your movement. A shared frame of reference.':tab==='calibration'?'Establish a stable skeleton and a neutral starting pose.':tab==='sessions'?'Capture tracking data for review and reproducible testing.':'Understand the signal before it reaches the robot.'}</p></div><div className="heading-actions"><button className={`button ${recording?'recording':''}`} disabled={!canAct||pending} onClick={()=>void command('record')} title={state?.demo?'Start the launcher with --camera to record':'Record tracking data (R)'}>{recording?<Square size={14}/>:<Circle size={14}/>}<span>{recording?'Stop recording':'Record'}</span></button><button className="button primary" onClick={()=>setModal('calibrate')}><ScanLine size={15}/>Calibrate<kbd>C</kbd></button></div></div>
        {(!connected||state?.status==='error'||error)&&<div className="banner error" role="status"><Unplug size={16}/><span>{error||state?.message||'Connecting to the local engine… The workspace reconnects automatically.'}</span></div>}
        {state?.demo&&<div className="demo-note"><span className="dot amber"/><span>Demonstration workspace</span><span className="demo-detail">Illustrative robot pose. No camera or physical robot connected.</span><button onClick={()=>setTab('diagnostics')}>Connection details<ArrowUpRight size={13}/></button></div>}
        {tab==='workspace'&&<>
          <div className={`workspace ${focus?'focused':''}`}>
            <section className="scene-panel" aria-label="Robot digital twin"><div className="panel-toolbar"><div className="panel-title"><Box size={16}/><h2>Digital twin</h2><span className="subtle-tag">SIMULATED</span></div><button className="icon-button" aria-label={focus?'Show camera panel':'Focus digital twin'} title="Focus view (V)" onClick={()=>setFocus(!focus)}><Focus size={16}/></button></div>
              <div className="scene-container"><Suspense fallback={<div className="scene-placeholder">Loading 3D workspace…</div>}><Scene state={state} view={view} grid={prefs.grid} targets={prefs.targets} body={prefs.body} reset={reset}/></Suspense>
                {!state?.arms&&<div className="scene-empty"><Box size={26}/><strong>Waiting for robot state</strong><span>The digital twin appears when the engine is ready.</span></div>}
                <div className="scene-topline"><span><span className="legend-line left"/>Left arm</span><span><span className="legend-line right"/>Right arm</span></div>
                <div className="scene-bottom"><span className="scene-axis"><b>X</b><b>Y</b><b>Z</b><span>metres</span></span><span>Drag to orbit · Scroll to zoom</span></div>
              </div>
              <div className="scene-controls"><div className="segmented" aria-label="3D camera view">{(['Perspective','Front','Top','Side'] as View[]).map(v=><button key={v} aria-pressed={view===v} onClick={()=>setView(v)}>{v}</button>)}</div><div className="scene-tools"><button className={`icon-button ${prefs.grid?'active':''}`} aria-label="Toggle floor grid" aria-pressed={prefs.grid} onClick={()=>toggle('grid')} title="Floor grid"><Grid2X2 size={16}/></button><button className={`icon-button ${prefs.targets?'active':''}`} aria-label="Toggle target pose" aria-pressed={prefs.targets} onClick={()=>toggle('targets')} title="Target pose"><Layers size={16}/></button><button className="icon-button" aria-label="Reset 3D camera" onClick={()=>setReset(reset+1)} title="Reset camera"><RotateCcw size={15}/></button></div></div>
            </section>
            {!focus&&<div className="right-column"><CameraPanel state={state} connected={connected}/><section className="tracking-panel"><div className="panel-toolbar"><div className="panel-title"><Activity size={16}/><h2>Tracking signal</h2></div><span className={`status-text ${state?.detected?'good':''}`}>{state?.detected?'Detected':'No pose'}</span></div><div className="tracking-content"><div className="quality-line"><span>Landmarks visible</span><strong>{state?.detected?`${quality}%`:'—'}</strong></div><div className="quality-track"><div style={{width:`${quality}%`}}/></div><div className="signal-row"><span>Neutral pose</span><span>{state?.calibrated?<><Check size={12}/> Calibrated</>:'Not calibrated'}</span></div><div className="signal-row"><span>Hand gestures</span><span>{state?.gestures?.length?state.gestures.map(g=>g.replaceAll('_',' ')).join(' · '):'Waiting for hands'}</span></div><button className="text-action" onClick={()=>setTab('calibration')}>Set up your tracking<ArrowUpRight size={13}/></button></div></section></div>}
          </div>
          <JointInspector state={state}/>
        </>}
        {tab==='calibration'&&<section className="content-surface"><div className="section-intro"><ScanLine size={24}/><h2>A consistent starting point</h2><p>Face the camera with both shoulders, elbows and hands in view. Complete skeleton measurement, then capture your neutral pose.</p></div><div className="setup-steps">{[['01','Measure your skeleton','Follow four poses: T-pose, bent elbows, arms up, arms down. Python measures and saves your bone lengths.','skeleton','Begin measurement'],['02','Capture neutral pose','Hold both arms steady for at least 0.8 seconds. This becomes the reference for relative joint angles.','calibrate','Capture neutral pose'],['03','Save your reference','Save the neutral calibration locally, so it can be loaded in your next session.','save_calibration','Save calibration']].map(([n,title,description,cmd,label])=><div className="setup-step" key={n}><span className="step-number">{n}</span><div><h3>{title}</h3><p>{description}</p></div><button className="button" disabled={!canAct||pending||((cmd==='skeleton'||cmd==='calibrate')&&!state?.detected)} onClick={()=>void command(cmd)}>{label}<ChevronRight size={14}/></button></div>)}</div><div className="calibration-feedback" role="status"><Crosshair size={17}/>{state?.skeleton_status||state?.message||'Ready when you are. Calibration instructions appear here.'}</div><div className="section-footer"><button className="button quiet" disabled={!canAct||pending} onClick={()=>void command('load_calibration')}><ArrowDownToLine size={15}/>Load saved calibration</button><button className="button quiet" disabled={!canAct||pending} onClick={()=>void command('reset_calibration')}><RotateCcw size={15}/>Reset neutral pose</button>{state?.skeleton_status&&<button className="button" disabled={!canAct||pending} onClick={()=>void command('skeleton')}>Cancel measurement</button>}</div></section>}
        {tab==='sessions'&&<section className="content-surface"><div className="section-intro"><Video size={24}/><h2>{recording?'Recording in progress':'Make your next test repeatable'}</h2><p>Record the existing Python tracking data as CSV. Capture positions, joint angles and gestures for analysis after your session.</p></div><div className="record-well"><span className={`record-indicator ${recording?'on':''}`}><Circle size={24}/></span><h3>{recording?'Session is being captured':'No recording in progress'}</h3><p>{state?.demo?'Connect the real camera to start recording.':'Files are saved in the configured recordings directory.'}</p><button className={`button ${recording?'danger':'primary'}`} disabled={!canAct||pending} onClick={()=>void command('record')}>{recording?<Square size={15}/>:<Circle size={15}/>} {recording?'Stop recording':'Start recording'}</button></div><div className="section-footer muted"><ListFilter size={15}/>This view controls the current session. Existing CSV files remain available in the recordings folder.</div></section>}
        {tab==='diagnostics'&&<section className="content-surface"><div className="section-intro"><Gauge size={24}/><h2>Signal & connection</h2><p>Live values reported by the Python engine. Camera FPS describes the application loop, not measured sensor-to-screen latency.</p></div><div className="diagnostic-grid"><dl>{[['Connection',connected?'Connected':'Offline'],['Source',state?.source||'Not available'],['Operating mode',mode],['Tracking',tracking],['Frame rate',state?.fps?`${state.fps.toFixed(1)} fps`:'Not available'],['Frame age',connected?`${state?.age_ms} ms`:'Stale / unavailable'],['Source resolution',state?.resolution?.join(' × ')||'Not available'],['Telemetry version',String(state?.schema_version||'—')]].map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl><div className="diagnostic-note"><Unplug size={22}/><h3>Physical robots are disconnected</h3><p>This web launcher runs the simulation backend. The current desktop launcher retains the hardware connection and commissioning workflows.</p><hr/><h3>Coordinates</h3><p>Body frame in metres: X right, Y forward, Z up. Depth comes from the tracking model. Mirroring changes presentation only.</p><button className="button" onClick={()=>{const blob=new Blob([JSON.stringify({...state,token:undefined},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='motion-twin-snapshot.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}}><ArrowDownToLine size={15}/>Export snapshot</button></div></div></section>}
      </main>
      <footer className="statusbar"><div><span className={`dot ${ready?'green':''}`}/>{ready?'Engine ready':'Waiting for engine'}<span className="status-divider"/>{state?.demo?'Demo data':tracking}</div><div><span>{state?.fps?`${state.fps.toFixed(1)} fps`:'— fps'}</span><span className="status-divider"/><span>{recording?'Recording CSV':'Recording idle'}</span><span className="status-divider"/><span>Local only</span></div></footer>
    </div>
    {notice&&<div className="toast" role="status"><Radio size={16}/><span>{notice}</span><button className="icon-button" aria-label="Dismiss notification" onClick={()=>setNotice('')}><X size={15}/></button></div>}
    {modal&&<Dialog title={modal==='settings'?'View preferences':modal==='help'?'Keyboard shortcuts':'Calibrate your reference'} onClose={()=>setModal(null)}>
      {modal==='settings'&&<><p className="dialog-description">Personal display settings. These do not change tracking or robot limits.</p>{([['grid','Floor grid','A spatial reference for height and reach.'],['targets','Target pose','Compare the commanded and simulated robot poses.'],['body','Tracked operator','Show the body skeleton beside the robot pair.']] as const).map(([key,label,description])=><label className="preference" key={key}><span><strong>{label}</strong><small>{description}</small></span><input type="checkbox" checked={prefs[key]} onChange={()=>toggle(key)}/></label>)}</>}
      {modal==='help'&&<div className="shortcuts">{[['C','Open calibration'],['R','Toggle CSV recording'],['V','Focus / restore 3D workspace'],['?','Show keyboard shortcuts'],['Esc','Close dialog'],['Tab','Move keyboard focus']].map(([key,label])=><div key={key}><span>{label}</span><kbd>{key}</kbd></div>)}</div>}
      {modal==='calibrate'&&<><div className="calibration-illustration"><ScanLine size={44} strokeWidth={1}/></div><h3>Hold your neutral pose</h3><p className="dialog-description">Keep both shoulders, elbows and hands visible. Hold still for at least 0.8 seconds before capturing.</p><div className="inline-status"><span className={`dot ${state?.detected?'green':'amber'}`}/>{state?.demo?'Demo is read-only. Launch with --camera to calibrate.':tracking}</div><div className="dialog-actions"><button className="button" onClick={()=>{setTab('calibration');setModal(null);}}>Skeleton setup</button><button className="button primary" disabled={!canAct||!state?.detected||pending} onClick={()=>{void command('calibrate');setModal(null);}}><Crosshair size={15}/>Capture pose</button></div></>}
    </Dialog>}
  </div>;
}

function CameraPanel({state,connected}:{state:Snapshot|null;connected:boolean}){
  const [url,setUrl]=useState('');const [failed,setFailed]=useState(false);
  useEffect(()=>{if(state?.demo||!connected||state?.status!=='running'){setUrl('');return;}
    let disposed=false;let current='';let timer:ReturnType<typeof setTimeout>;const controller=new AbortController();
    const update=async()=>{try{const response=await fetch('/api/v1/frame',{signal:controller.signal});if(!response.ok||response.status===204)throw new Error('No frame');const blob=await response.blob();if(disposed)return;const next=URL.createObjectURL(blob);setUrl(next);setFailed(false);if(current)URL.revokeObjectURL(current);current=next;}catch{if(!disposed)setFailed(true);}finally{if(!disposed)timer=setTimeout(update,100);}};void update();return()=>{disposed=true;controller.abort();clearTimeout(timer);if(current)URL.revokeObjectURL(current);};
  },[state?.demo,state?.status,connected]);
  return <section className="camera-panel"><div className="panel-toolbar"><div className="panel-title"><Camera size={16}/><h2>Camera</h2></div><span className="subtle-tag">{state?.demo?'OFFLINE':connected?'LIVE':'OFFLINE'}</span></div><div className="camera-image">{url&&!failed?<img src={url} alt="Live camera feed with Python tracking overlay"/>:<div className="camera-empty"><span className="camera-outline"><Camera size={27} strokeWidth={1.3}/></span><strong>{state?.demo?'Camera preview':'Waiting for camera'}</strong><p>{state?.demo?'Your live feed will appear here.':'Check the local tracking engine.'}</p><span>{state?.demo?'Start with --camera to connect':'Reconnecting automatically'}</span></div>}<div className="camera-corner top-left"/><div className="camera-corner bottom-right"/></div><div className="camera-footer"><span>{state?.source||'No source'}</span><span>{!state?.demo&&state?.resolution?state.resolution.join(' × '):'RGB camera'}</span></div></section>;
}

function JointInspector({state}:{state:Snapshot|null}){
  const [side,setSide]=useState('left'); const arm=state?.arms?.[side];
  return <section className="joint-inspector"><div className="inspector-heading"><div className="panel-title"><SlidersHorizontal size={15}/><h2>Joint inspector</h2></div><div className="segmented compact">{['left','right'].map(s=><button key={s} aria-pressed={s===side} onClick={()=>setSide(s)}>{s==='left'?'Left arm':'Right arm'}</button>)}</div><span className="muted">Simulated position · degrees</span></div><div className="joint-values">{['base','shoulder','elbow','wrist_1','wrist_2','wrist_3'].map((joint,i)=><div className="joint-value" key={joint}><span><small>J{i+1}</small>{joint.replace('_',' ')}</span><strong>{arm?.joints_deg[joint]?.toFixed(1)??'—'}<small>°</small></strong><div className="joint-track"><i style={{left:`${Math.max(0,Math.min(100,((arm?.joints_deg[joint]??0)+180)/360*100))}%`}}/></div></div>)}</div></section>;
}

function Dialog({title,onClose,children}:{title:string;onClose:()=>void;children:ReactNode}){
  const dialog=useRef<HTMLDialogElement>(null);
  useEffect(()=>{const previous=document.activeElement as HTMLElement;dialog.current?.showModal();return()=>{dialog.current?.close();previous?.focus();};},[]);
  return <dialog ref={dialog} onCancel={onClose} onClick={e=>{if(e.target===e.currentTarget)onClose();}} aria-labelledby="dialog-title"><div className="dialog-content"><div className="dialog-heading"><h2 id="dialog-title">{title}</h2><button className="icon-button" aria-label="Close dialog" onClick={onClose}><X size={18}/></button></div>{children}</div></dialog>;
}
