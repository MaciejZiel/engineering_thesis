import {useEffect, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Point, Snapshot, View} from './types';

type Props = {state: Snapshot | null; view:View; grid:boolean; targets:boolean; body:boolean; reset:number};
const surface = '#9aa6ae';
const toScene = (p:number[], mirror=false) => new THREE.Vector3(p[0] * (mirror ? -1 : 1), p[2], -p[1]);

export default function Scene({state,view,grid,targets,body,reset}:Props) {
  const poseSignature=JSON.stringify([state?.arms,state?.body_m,state?.mirrored]);
  const element = useRef<HTMLDivElement>(null);
  const engine = useRef<{root:THREE.Group; camera:THREE.PerspectiveCamera; controls:OrbitControls; grid:THREE.GridHelper} | null>(null);
  const [error,setError] = useState('');
  useEffect(() => {
    const container = element.current!;
    let renderer:THREE.WebGLRenderer;
    try {renderer = new THREE.WebGLRenderer({antialias:true,alpha:true});} catch {setError('3D rendering is unavailable. Enable hardware acceleration, or use the joint inspector.'); return;}
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    renderer.setClearColor('#171a1d', 1);
    container.appendChild(renderer.domElement);
    renderer.domElement.setAttribute('aria-label','Interactive 3D workspace. Drag to orbit; scroll to zoom.');
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight('#dae9f6','#39404a',3));
    const light = new THREE.DirectionalLight('#fff3df',4); light.position.set(1,4,3); scene.add(light);
    const camera = new THREE.PerspectiveCamera(34,1,0.01,100);
    const controls = new OrbitControls(camera,renderer.domElement);
    controls.enableDamping = false; controls.minDistance=1; controls.maxDistance=7; controls.maxPolarAngle=Math.PI*.49;
    const gridHelper = new THREE.GridHelper(4,20,'#48515a','#293139'); gridHelper.position.y=-.005; scene.add(gridHelper);
    const root = new THREE.Group(); scene.add(root);
    engine.current={root,camera,controls,grid:gridHelper};
    const resize = new ResizeObserver(() => {const w=container.clientWidth,h=container.clientHeight; renderer.setSize(w,h); camera.aspect=w/Math.max(h,1);camera.updateProjectionMatrix();}); resize.observe(container);
    let id=0; let last=0;
    const draw=(time:number)=>{id=requestAnimationFrame(draw); if(document.hidden || time-last<30)return;last=time;controls.update();renderer.render(scene,camera);};id=requestAnimationFrame(draw);
    return()=>{cancelAnimationFrame(id);resize.disconnect();controls.dispose();dispose(root);gridHelper.geometry.dispose();(gridHelper.material as THREE.Material).dispose();renderer.dispose();container.removeChild(renderer.domElement);engine.current=null;};
  },[]);
  useEffect(()=>{
    const e=engine.current;if(!e)return;
    const positions:Record<View,Point>={Perspective:[1.6,1.35,2.4],Front:[0,.6,3],Top:[0,3,.001],Side:[3,.6,0]};
    e.camera.position.set(...positions[view]);e.controls.target.set(0,.42,0);e.controls.update();
  },[view,reset]);
  useEffect(()=>{if(engine.current)engine.current.grid.visible=grid;},[grid]);
  useEffect(()=>{
    const e=engine.current;if(!e)return;dispose(e.root);e.root.clear();
    const mirror=state?.mirrored ?? true;
    const tube=(a:THREE.Vector3,b:THREE.Vector3,r:number,color:string,opacity=1)=>{
      const delta=b.clone().sub(a);if(delta.length()<.0001)return;
      const mesh=new THREE.Mesh(new THREE.CylinderGeometry(r,r,delta.length(),16),new THREE.MeshStandardMaterial({color,metalness:.55,roughness:.4,transparent:opacity<1,opacity}));
      mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());e.root.add(mesh);
    };
    const ball=(p:THREE.Vector3,r:number,color:string)=>{const mesh=new THREE.Mesh(new THREE.SphereGeometry(r,18,12),new THREE.MeshStandardMaterial({color,metalness:.35,roughness:.32}));mesh.position.copy(p);e.root.add(mesh);};
    for(const [side,arm] of Object.entries(state?.arms ?? {})){
      const color=side==='left'?'#dba36f':'#79a7bf';
      const points=arm.points_m.map(p=>toScene(p,mirror));
      points.slice(1).forEach((p,i)=>tube(points[i],p,i<3?.038:.026,surface));
      points.slice(1,-1).forEach(p=>ball(p,.045,color));
      if(points.length){tube(points[0],points[0].clone().add(new THREE.Vector3(0,.035,0)),.085,'#6e7880');ball(points.at(-1)!, .024,'#dce3e6');}
      if(targets)arm.target_points_m.slice(1).forEach((p,i)=>tube(toScene(arm.target_points_m[i],mirror),toScene(p,mirror),.012,color,.22));
    }
    if(body && state?.body_m?.length){
      const points=state.body_m; const at=(i:number)=>toScene(points[i],mirror).add(new THREE.Vector3(-1.0,1.05,-.3));
      for(const [a,b] of [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]]){
        if(points[a]?.[3]>.5&&points[b]?.[3]>.5){tube(at(a),at(b),.012,'#8fbdab');ball(at(b),.022,'#8fbdab');}
      }
      if(points[0]?.[3]>.5)ball(at(0),.07,'#8fbdab');
    }
  },[poseSignature,targets,body]);
  return <div ref={element} className="three-scene">{error&&<div className="scene-error" role="alert">{error}</div>}</div>;
}

function dispose(group:THREE.Group){group.traverse(object=>{if(object instanceof THREE.Mesh){object.geometry.dispose();const materials=Array.isArray(object.material)?object.material:[object.material];materials.forEach(m=>m.dispose());}});}
