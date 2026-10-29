"""Native diagrams.net XML and vector exports. Default preserves XML edits.
Use --rebuild-layouts to regenerate initial layouts. Export arbitrary new editor
shapes in diagrams.net; this renderer supports the original draft's subset.
"""
from pathlib import Path
import argparse, copy, csv, json, os, subprocess
import xml.etree.ElementTree as E
os.environ.setdefault('MPLCONFIGDIR','/tmp/ral-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyBboxPatch, FancyArrowPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parents[1]
OUT=R/'figures'; EDIT=OUT/'editable'
INK='#243447'; MUTED='#526377'; BLUE='#28658C'; PB='#EDF4FA'
TEAL='#147D77'; PT='#E8F5F1'; ORANGE='#B66A1B'; PO='#FFF2DF'
PURPLE='#7153A1'; PP='#F1ECF8'; RED='#B54148'; GRAY='#E2E7ED'; LIGHT='#F7F9FC'
class D:
 def __init__(self,name,title,w,h):
  self.name=name; self.n=1; self.boxes={}
  self.file=E.Element('mxfile',host='app.diagrams.net',type='device',compressed='false')
  page=E.SubElement(self.file,'diagram',id=name,name=title)
  model=E.SubElement(page,'mxGraphModel',dx=str(w),dy=str(h),grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(w),pageHeight=str(h),math='0',shadow='0')
  self.root=E.SubElement(model,'root'); E.SubElement(self.root,'mxCell',id='0'); E.SubElement(self.root,'mxCell',id='1',parent='0')
 def box(self,x,y,w,h,text='',fill='white',stroke=INK,fs=18,bold=False,rounded=True,align='center',color=INK,dash=False,ellipse=False):
  self.n+=1; identity=f'n{self.n}'
  style=f'rounded={int(rounded)};whiteSpace=wrap;html=0;fontFamily=Helvetica;fontSize={fs};fontColor={color};fontStyle={int(bold)};fillColor={fill};strokeColor={stroke};strokeWidth=1.3;align={align};verticalAlign=middle;spacing=3;'
  if ellipse: style+='ellipse;'
  if dash: style+='dashed=1;'
  cell=E.SubElement(self.root,'mxCell',id=identity,value=text,style=style,vertex='1',parent='1')
  E.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
  self.boxes[identity]=(x,y,w,h); return identity
 def text(self,x,y,w,h,text,fs=17,bold=False,color=INK,align='center'):
  return self.box(x,y,w,h,text,fill='none',stroke='none',fs=fs,bold=bold,rounded=False,align=align,color=color)
 def line(self,points,color=INK,width=1.5,arrow=False,dash=False,source=None,target=None):
  self.n+=1
  style=f'edgeStyle=none;html=0;strokeColor={color};strokeWidth={width};endArrow={"block" if arrow else "none"};endFill=1;'
  if dash: style+='dashed=1;'
  attrs=dict(id=f'e{self.n}',edge='1',parent='1')
  for key,ident,point in [('source',source,points[0]),('target',target,points[-1])]:
   if ident:
    attrs[key]=ident; x,y,w,h=self.boxes[ident]; pre='exit' if key=='source' else 'entry'
    style+=f'{pre}X={(point[0]-x)/w:g};{pre}Y={(point[1]-y)/h:g};{pre}Perimeter=0;'
  attrs['style']=style; cell=E.SubElement(self.root,'mxCell',**attrs)
  geo=E.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
  for point,kind in [(points[0],'sourcePoint'),(points[-1],'targetPoint')]:
   E.SubElement(geo,'mxPoint',x=str(point[0]),y=str(point[1]),attrib={'as':kind})
  if len(points)>2:
   arr=E.SubElement(geo,'Array',attrib={'as':'points'})
   for x,y in points[1:-1]: E.SubElement(arr,'mxPoint',x=str(x),y=str(y))
 def save(self):
  E.indent(self.file); E.ElementTree(self.file).write(EDIT/f'{self.name}.xml',encoding='utf-8',xml_declaration=True)
def grip(d,x,y,c=TEAL,closed=False):
 d.line([(x,y),(x,y+16),(x-13,y+16),(x-13,y+33)],c,2.5)
 d.line([(x,y+16),(x+13,y+16),(x+13,y+33)],c,2.5)
 if closed:
  d.line([(x-13,y+33),(x-5,y+33)],c,2.5); d.line([(x+13,y+33),(x+5,y+33)],c,2.5)
def overview():
 d=D('pipeline_overview','01 Pipeline overview',1000,302)
 d.text(12,4,976,26,'Adapt a pretrained VLA to a new action interface',22,True)
 for x,c,bg,t in [(12,TEAL,PT,'1  Dense skill acquisition'),(348,ORANGE,PO,'2  Binary refinement + SIL'),(684,PURPLE,PP,'3  Full-task composition')]:
  d.box(x,46,304,207,fill=bg,stroke=c); d.text(x+10,54,284,28,t,18,True,c)
 d.line([(318,147),(343,147)],arrow=True,width=2); d.line([(654,147),(679,147)],arrow=True,width=2)
 for i,(label,x) in enumerate([('Approach',54),('Pickup',150),('Place',246)]):
  grip(d,x,101,closed=i==1); d.box(x-8,145 if i!=1 else 132,16,16,fill=TEAL,stroke=TEAL,rounded=False); d.text(x-42,169,84,22,label,16)
 d.line([(28,162),(298,162)],TEAL,1); d.text(24,197,280,44,'Progress rewards\nSkill-specific starts',17,color=TEAL)
 d.box(365,100,116,69,'Skill success\n0 / 1',stroke=ORANGE,fs=18)
 for dx,dy in [(8,-6),(4,-3),(0,0)]: d.box(519+dx,105+dy,111,58,stroke=ORANGE)
 d.text(523,111,108,45,'Successful\ntrajectory bank',13,color=ORANGE); d.line([(481,134),(515,134)],ORANGE,arrow=True)
 d.text(360,189,280,46,'Balance earlier + new skills\nConsolidate one policy',17,color=ORANGE)
 for i,label in enumerate(['Reach','Grasp','Lift','Move','Place']):
  x=705+i*55; d.box(x,105,39,39,str(i+1),stroke=PURPLE,ellipse=True,fs=18); d.text(x-5,151,50,22,label,14)
  if i<4: d.line([(x+39,124),(x+53,124)],PURPLE,arrow=True)
 d.text(696,191,280,44,'One full-task instruction\nMilestone credit during RL',17,color=PURPLE)
 d.text(15,263,970,27,'Evaluation: empty-gripper start  |  one policy  |  strict complete-task success',18,True); d.save()
def architecture():
 d=D('smolvla_residual_lora','02 SmolVLA residual and LoRA',1000,433)
 d.box(167,40,469,105,fill=PB,stroke=BLUE); d.text(177,45,250,24,'Pretrained SmolVLA',19,True,BLUE)
 images=d.box(12,82,130,53,'Camera images',fs=15)
 tokens=d.box(190,82,190,53,'Vision-language\nencoder + connector',stroke=BLUE,fs=17)
 instruction=d.box(12,154,130,44,'Instruction',fs=18)
 state_in=d.box(12,244,130,53,'Proprioception\n6d state input',fs=14)
 expert=d.box(437,82,169,53,'Action expert',stroke=BLUE,fs=18)
 lora=d.box(451,45,142,27,'+ LoRA adapters',fill=PO,stroke=ORANGE,fs=14,bold=True)
 prior=d.box(673,82,144,53,'Map to target\nprior p: 8 x 5',fill=PB,stroke=BLUE,fs=17)
 d.line([(142,108),(190,108)],arrow=True,source=images,target=tokens)
 d.line([(142,176),(158,176),(158,121),(190,121)],arrow=True,source=instruction,target=tokens)
 d.line([(380,108),(437,108)],BLUE,arrow=True,source=tokens,target=expert)
 d.line([(522,72),(522,82)],ORANGE,arrow=True,source=lora,target=expert)
 d.line([(606,108),(673,108)],BLUE,arrow=True,source=expert,target=prior)
 project=d.box(190,174,190,52,'Frozen token projection\n512d visual features',fill=PB,stroke=BLUE,fs=16)
 obs=d.box(190,244,190,61,'Observation x\n512 + 6 = 518d',fill=PT,stroke=TEAL,fs=18,bold=True)
 d.line([(285,135),(285,174)],BLUE,arrow=True,source=tokens,target=project)
 # Gap in the visual-feature edge marks a crossing, not a junction.
 d.box(281,152,8,8,fill='white',stroke='none',ellipse=True)
 d.line([(142,270),(166,270),(166,156),(522,156),(522,135)],arrow=True,source=state_in,target=expert)
 d.line([(285,226),(285,244)],BLUE,arrow=True,source=project,target=obs)
 d.line([(142,270),(190,270)],arrow=True,source=state_in,target=obs)
 d.box(164,268,4,4,fill=INK,stroke=INK,ellipse=True)
 res=d.box(633,231,183,80,'Residual MLP\n558 -> 1024 -> 1024\n-> 40; bounded tanh',fill=PO,stroke=ORANGE,fs=14,bold=True)
 d.line([(380,270),(633,270)],TEAL,arrow=True,source=obs,target=res)
 d.line([(745,135),(745,231)],BLUE,arrow=True,source=prior,target=res); d.text(751,167,105,48,'Prior input\nflatten: 40d',14,color=BLUE)
 add=d.box(866,245.5,49,49,'+',ellipse=True,fs=26)
 d.line([(816,270),(866,270)],ORANGE,arrow=True,source=res,target=add)
 d.line([(817,108),(890.5,108),(890.5,245.5)],BLUE,arrow=True,source=prior,target=add)
 output=d.box(848,325,137,49,'tanh: 8 x 5\nexecute 4 slots',fill=PT,stroke=TEAL,fs=17)
 d.line([(890.5,294.5),(890.5,325)],arrow=True,source=add,target=output)
 d.text(192,343,624,29,'Pretrained action prior + bounded correction',19,True)
 d.line([(12,389),(988,389)],'#CAD3DD'); d.text(12,398,976,25,'Earlier phases: expert LoRA + residual    |    Full-task continuation: VLA / LoRA frozen; residual trained',17); d.save()
def phase():
 d=D('phase_feedback','04 Feedback resolution',490,307); d.text(8,4,474,24,'Feedback at different resolutions',20,True)
 for i,label in enumerate(['Reach','Grasp','Lift','Move','Place']):
  x=132+i*68; d.box(x,40,64,29,label,fill=LIGHT,stroke='#CCD5DF',fs=14)
  if i<4: d.line([(x+64,54),(x+68,54)],MUTED,arrow=True)
 for y,label in [(98,'Terminal only'),(144,'Phase only'),(190,'Dense')]:
  d.text(8,y-8,122,27,label,14,True,align='left'); d.line([(132,y+7),(472,y+7)],'#CBD4DE')
 d.box(456,92,13,13,fill=RED,stroke=RED,ellipse=True)
 for i in range(5): d.box(180+i*68,138,10,10,fill=PURPLE,stroke=PURPLE,rounded=False)
 for i in range(5):
  d.box(180+i*68,190,10,10,fill=PURPLE,stroke=PURPLE,rounded=False)
  for j in range(3):
   x=140+i*68+j*15; d.line([(x,196),(x,187-j*4)],TEAL,width=2)
 d.box(8,230,474,62,'Progress from zero:  0.4 -> 0.7 -> 0.5 -> 0.7 -> 0.8\nNew progress credit:  +0.4   +0.3      0        0      +0.1',fill=PT,stroke=TEAL,fs=15); d.save()
def credit():
 d=D('milestone_credit','03 Milestone credit',490,307); d.text(8,4,474,25,'Successful prefix, failed placement',19,True)
 xs=[125,247,369]
 for i,(label,ok) in enumerate([('Approach',True),('Held pickup',True),('Placement',False)]):
  x=xs[i]; d.box(x,48,108,45,label,fill=PT if ok else '#FCEEEF',stroke=TEAL if ok else RED,fs=16); d.text(x,98,108,24,'achieved' if ok else 'failed',17,True,TEAL if ok else RED)
 for y,lab in [(148,'Local credit'),(196,'Failure bonus'),(244,'Used credit')]: d.text(7,y-3,117,29,lab,13,True,align='left')
 for i,x in enumerate(xs):
  label='positive' if i<2 else 'negative'; d.box(x,145,108,29,label,fill=LIGHT,stroke='#CAD3DD',fs=17)
  d.box(x,193,108,29,'blocked' if i<2 else 'no duplicate',fill=PO if i<2 else LIGHT,stroke=ORANGE if i<2 else '#CAD3DD',fs=13)
  d.box(x,241,108,29,label,fill=PT if i<2 else '#FCEEEF',stroke=TEAL if i<2 else RED,fs=17)
 d.text(8,277,474,22,'Example with mixed outcomes in each milestone group.',14,color=MUTED); d.save()
def retention():
 data=json.loads((R/'evidence/skill_retention_rounds.json').read_text()); d=D('skill_retention','05 Skill forgetting and recovery',1000,353)
 d.text(8,4,474,25,'(a) Reaching lost during placement training',18,True); d.text(511,4,481,25,'(b) Balanced SIL recovers earlier behavior',20,True)
 y0=267; scale=2.48
 for left,right in [(52,473),(548,979)]:
  for tick in [0,20,40,60,80]:
   y=y0-tick*scale; d.line([(left,y),(right,y)],'#DDE3EB',width=.8); d.text(left-43,y-11,36,22,str(tick),15,color=MUTED,align='right')
  d.line([(left,68),(left,y0),(right,y0)],MUTED,width=1)
 d.text(8,32,470,24,'Reaching success (%)',16,color=MUTED); d.text(511,32,470,24,'Skill success (%)',16,color=MUTED)
 names=['Retained\nreaching','After\nplacement RL','After placement-\nonly imitation']
 for j,(_,rates) in enumerate(data['forgetting'].items()):
  val=sum(rates)/len(rates)*100; x=88+j*138; c=[BLUE,RED,RED][j]
  d.box(x,y0-val*scale,57,val*scale,fill=c,stroke=c,rounded=False); d.text(x-10,y0-max(rates)*100*scale-29,77,24,f'{val:.1f}',17,True)
  for k,v in enumerate(rates): d.box(x+16+k*10,y0-v*100*scale-2,4,4,stroke=INK,ellipse=True)
  d.text(x-34,276,130,44,names[j],14)
 for j,(skill,conditions) in enumerate(data['recovery'].items()):
  center=616+j*139
  for k,(_,rates) in enumerate(conditions.items()):
   val=sum(rates)/len(rates)*100; x=center-45+k*41; c=GRAY if k==0 else TEAL
   d.box(x,y0-val*scale,34,val*scale,fill=c,stroke=MUTED if k==0 else TEAL,rounded=False); d.text(x-13+(-5 if k==0 else 5),y0-max(rates)*100*scale-27,60,22,f'{val:.1f}',16,True)
   for n,v in enumerate(rates): d.box(x+7+n*8,y0-v*100*scale-2,4,4,stroke=INK,ellipse=True)
  d.text(center-61,276,122,40,skill.replace(' ','\n'),15)
 d.box(665,66,14,13,fill=GRAY,stroke=MUTED,rounded=False); d.text(684,60,92,25,'Before SIL',15,align='left')
 d.box(786,66,14,13,fill=TEAL,stroke=TEAL,rounded=False); d.text(805,60,130,25,'After SIL',15,align='left')
 d.line([(495,41),(495,320)],'#CAD3DD',dash=True); d.text(8,326,984,22,'Bars: mean success. Marks: evaluation rounds. Forgetting and recovery are separate experiments.',15,color=MUTED); d.save()
def wx200():
 d=D('wx200_slots','06 WX200 illustration slots',1000,287); d.text(10,4,980,26,'WX200: five-DoF embodiment extension',22,True)
 d.box(10,44,303,224,fill=LIGHT,stroke=MUTED,dash=True); d.text(20,53,283,28,'(a) Apparatus + control axes',17,True)
 d.text(22,95,279,120,'Insert actual robot photograph\nor verified CAD illustration\n\nLabel 5 joints, workspace,\ncamera views, object and target',16,color=MUTED)
 d.text(24,223,275,27,'Setup illustration to be added',16,color=MUTED)
 d.text(334,49,650,27,'(b) Full-task rollout: one instruction, empty-gripper start',19,True)
 for j,label in enumerate(['Reach','Grasp','Lift','Transport','Release']):
  x=334+j*132; d.box(x,93,122,132,fill=LIGHT,stroke=MUTED,dash=True); d.text(x+5,107,112,40,'Frame\nplaceholder',17,color=MUTED); d.text(x+5,174,112,22,label,17,True); d.text(x+5,199,112,21,'t = ___ s',16,color=MUTED)
 d.text(334,238,650,29,'Add instruction, trial ID and deployment condition with the real images.',17,color=MUTED); d.save()
def styles(s):
 result={}
 for entry in s.split(';'):
  if '=' in entry:
   k,v=entry.split('=',1); result[k]=v
  elif entry: result[entry]='1'
 return result
def render(path):
 plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 root=E.parse(path).getroot(); model=root.find('diagram/mxGraphModel'); w,h=float(model.get('pageWidth')),float(model.get('pageHeight'))
 fig=plt.figure(figsize=(w/100,h/100),facecolor='white'); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,w),ylim=(h,0)); ax.axis('off'); issues=[]
 for idx,cell in enumerate(model.findall('root/mxCell')):
  st=styles(cell.get('style','')); g=cell.find('mxGeometry')
  if g is None: continue
  z=1+idx/100; c=st.get('strokeColor',INK); lw=float(st.get('strokeWidth',1))*.72
  if cell.get('edge')=='1':
   points=[(float(p.get('x')),float(p.get('y'))) for p in [g.find("mxPoint[@as='sourcePoint']"),*g.findall('Array/mxPoint'),g.find("mxPoint[@as='targetPoint']")]]
   for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
    if a==b: continue
    if i==len(points)-2 and st.get('endArrow','none')!='none': ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,linewidth=lw,color=c,shrinkA=0,shrinkB=0,zorder=z,linestyle='--' if st.get('dashed')=='1' else '-'))
    else: ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=lw,zorder=z,linestyle='--' if st.get('dashed')=='1' else '-')
   continue
  x,y,bw,bh=[float(g.get(k,0)) for k in ['x','y','width','height']]
  if x<0 or y<0 or x+bw>w+.1 or y+bh>h+.1: issues.append(cell.get('id')+': outside canvas')
  kwargs=dict(facecolor=st.get('fillColor','white'),edgecolor=st.get('strokeColor',INK),linewidth=lw,zorder=z,linestyle='--' if st.get('dashed')=='1' else '-')
  if st.get('ellipse')=='1': patch=Ellipse((x+bw/2,y+bh/2),bw,bh,**kwargs)
  elif st.get('rounded')=='1': patch=FancyBboxPatch((x,y),bw,bh,boxstyle='round,pad=0,rounding_size=6',**kwargs)
  else: patch=Rectangle((x,y),bw,bh,**kwargs)
  ax.add_patch(patch); value=cell.get('value','')
  if not value: continue
  fs=float(st.get('fontSize',18))*.72; weight='bold' if st.get('fontStyle')=='1' else 'normal'; ha=st.get('align','center'); tx=x+3 if ha=='left' else (x+bw-3 if ha=='right' else x+bw/2)
  ax.text(tx,y+bh/2,value,fontsize=fs,fontfamily='DejaVu Sans',fontweight=weight,color=st.get('fontColor',INK),ha=ha,va='center',linespacing=1.2,parse_math=False,zorder=z+.001)
  fp=FontProperties(family='DejaVu Sans',weight=weight)
  for line in value.splitlines():
   if line:
    length=TextPath((0,0),line,size=fs,prop=fp).get_extents().width/.72
    if length>bw-5: issues.append(f'{cell.get("id")}: text width {length:.0f} > {bw}')
  if len(value.splitlines())*fs*1.2/.72>bh+2: issues.append(cell.get('id')+': text taller than box')
 for ext in ['pdf','svg','png']: fig.savefig(OUT/f'{path.stem}.{ext}',dpi=180)
 svg=OUT/f'{path.stem}.svg'
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)
 if issues: raise ValueError(f'{path.name}: '+', '.join(issues))
 return len(model.findall('root/mxCell'))-2
def combine():
 combined=E.Element('mxfile',host='app.diagrams.net',type='device',compressed='false')
 pages=[E.parse(p).getroot().find('diagram') for p in EDIT.glob('*.xml') if p.name!='ral_figures.xml']
 for page in sorted(pages,key=lambda p:p.get('name')): combined.append(copy.deepcopy(page))
 E.indent(combined); E.ElementTree(combined).write(EDIT/'ral_figures.xml',encoding='utf-8',xml_declaration=True)
def render_supplied_svg(name):
 # The author's replacement SVG is authoritative; do not overwrite it from XML.
 source=OUT/f'{name}.svg'
 for ext in ['pdf','png']:
  args=['rsvg-convert','--format',ext,'--output',str(OUT/f'{name}.{ext}')]
  if ext=='png': args.extend(['--zoom','2'])
  subprocess.run([*args,str(source)],check=True)
def table():
 rows=list(csv.DictReader((R/'evidence/libero_paper_results_export_20261026.csv').open())); final=[r for r in rows if r['evaluation_stage']=='final']
 labels={'Terminal only lambda=5':r'Terminal only, $\lambda=5$','Primary dense lambda=5':r'Dense + phase, $\lambda=5$','Mixed-outcome filtered dense lambda=5':r'Filtered dense + phase, $\lambda=5$','Dense shaping lambda=1':r'Dense + phase, $\lambda=1$','Phase shaping lambda=1':r'Phase-only shaping, $\lambda=1$','Terminal only':'Terminal only','Dense reward':'Dense mixture'}
 lines=[r'\begin{table}[t]',r'\caption{LIBERO reward comparison. $\lambda$ is the terminal verifier coefficient; success and $\Delta$ are percentages and percentage-point gains. The low-success diagnostic tests initial learning, without a phase-only condition.}',r'\label{tab:libero}\centering\small',r'\begin{tabular}{@{}lrrr@{}}\toprule',r'Configuration & Updates & Success & $\Delta$\\\midrule',r'\multicolumn{4}{@{}l}{LIBERO-Spatial: initial 61.0\%}\\']
 for series in ['full_suite_reward_ablation','single_task_low_success']:
  if series=='single_task_low_success': lines.extend([r'\midrule',r'\multicolumn{4}{@{}l}{Low-success diagnostic: initial 2.0\% (1,000 rollouts)}\\'])
  for row in final:
   if row['series']==series: lines.append(f"{labels[row['configuration']]} & {row['rl_update_step']} & {float(row['success_percent']):.1f} & {float(row['change_from_initial_pp']):+.1f}"+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}']); (R/'tables/libero_results.tex').write_text('\n'.join(lines)+'\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--rebuild-layouts',action='store_true'); args=parser.parse_args(); EDIT.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
 if args.rebuild_layouts:
  for fn in [overview,architecture,phase,credit,retention,wx200]: fn()
 paths=[p for p in sorted(EDIT.glob('*.xml')) if p.name!='ral_figures.xml']
 if len(paths)!=6: raise ValueError(f'Expected six diagrams, got {len(paths)}')
 for p in paths:
  if p.stem=='skill_retention':
   render_supplied_svg(p.stem); print(f'{p.stem}.svg: exported supplied figure to PDF and PNG')
  else: print(f'{p.name}: rendered {render(p)} editable objects')
 combine(); table(); print('Rendered five native XML figures and the supplied retention SVG; Table II uses exact CSV endpoint rows.')
if __name__=='__main__': main()
