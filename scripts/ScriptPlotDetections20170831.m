%% description

% Script that reads in detection maps from CNN and does non-maximal
% suppression and connected components to output a list of
% detections. Older versions of this script are also in the repo. Note
% that this uses compiled versions of the MATLAB functions to run the
% code in parallel on the Janelia cluster.

%% paths
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling;
addpath /groups/branson/home/bransonk/codepacks/keller-lab-block-filetype/matlabWrapper/

datasetname = 'results';


switch datasetname,
  case 'results',
    % TODO: paths actually start with ~/, not /, must expand $HOME
    rawdatadir = '/data/division_detection/klb'
    preddatadir = '/data/division_detection/results/dense/'
    sparsepreddatadir = '/data/division_detection/results/sparse'
    savedir = '/data/division_detection/results/detections'
    % TODO: unclear what these are for
    inmatfilestr = '/nrs/branson/MouseLineaging/imregionalmaxdata_dataset1';
    inmatfilestr_conncomp = '/nrs/branson/MouseLineaging/conncompdata_dataset1';
  case 'dataset_1',
    rawdatadir = '/media/KellerS7/SV1/KM_16-06-16/Mmu_E1_TmCherryxH2BeGFP_20160616_155129.corrected/Results/TimeFused.Corrected/*';
    preddatadir = '/nrs/turaga/bergera/division_detection/prediction_outbox/mk5_large_small_batch_balanced_good_reweight/full_dataset_1/dense';
    sparsepreddatadir = '/nrs/branson/MouseLineaging/mk5_large_small_batch_balanced_good_reweight_dataset1_sparse';
    savedir = '/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/Detections_Dataset1_20170915';
    inmatfilestr = '/nrs/branson/MouseLineaging/imregionalmaxdata_dataset1';
    inmatfilestr_conncomp = '/nrs/branson/MouseLineaging/conncompdata_dataset1';
  case 'dataset_2',
    rawdatadir = '/media/KellerS8/SV1/KM_14-08-13/Mmu_E1_CAGTAG1_TrackingTest_0_20140813_104820.corrected/Results/TimeFused.Corrected';
    preddatadir = '/nrs/turaga/bergera/division_detection/prediction_outbox/mk5_large_small_batch_balanced_good_reweight/full_dataset_2/dense';
    sparsepreddatadir = '/nrs/branson/MouseLineaging/mk5_large_small_batch_balanced_good_reweight_dataset2_sparse';
    savedir = '/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/Detections_Dataset2_20170915';    inmatfilestr = '/nrs/branson/MouseLineaging/imregionalmaxdata_dataset2';
    inmatfilestr_conncomp = '/nrs/branson/MouseLineaging/conncompdata_dataset2';

  otherwise
    error('Unknown dataset');
end

% TODO: hard coded
gtdatafiles = {'/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/divisionAnnotations.mat'
  'AnnotatedTimePoints.csv'
  'TPs_DivAnnotations.csv'};

% TODO: hard coded
maxprojdatadir = '/media/KellerS8/Processing/SV1/14-05-21/DivisionDetection/Kristin/ProjectionsWavelet';
%maxprojdatadir = '/media/KellerS8/SV1/14-05-21/DivisionDetection/Kristin/Projections';
% some of the z-coordinates are in the isotropic coordinate system
gtratios = [1,1,1,1
  1,1,.2,1
  1,1,.2,1];

if ~exist(savedir,'dir'),
  mkdir(savedir);
end

iscompletedata = true;

% sudo mount //Keller-S8/Processing -t cifs -o uid=990313,gid=93319,username=SiMView /media/KellerS8/

%% parameters

% threshold raw scores here for efficiency
predthresh = .01;

% predfiletype = 'sparseh5';
% preddatasetname = '/coo';
% predshapename = '/shape';
predfiletype = 'h5';
preddatasetname = '/predictions';

rawfiletype = 'klb';
rawisregexp = true;
rawisrecursive = true;
rawmaxdepth = 1;
%rawfilestr
%=%'SPM00_TM(\d+)_CM00_CM01_CHN00.fusedStack.corrected.shifted.klb';

% TODO: is this string format correct? The expected format is two digit
% zeropadding
% eg Volume_15.klb , Volume_05.klb
rawfilestr = 'Volume_(\d+).klb';
poslabels = [1,2,3,4,5,103];

usecluster = false;

%% cluster parameters

username = GetUserName();
TMP_ROOT_DIR = fullfile('/scratch',username);
MCR_CACHE_ROOT = fullfile(TMP_ROOT_DIR,'mcr_cache_root');

% TODO: hardcoded
MCR = '/groups/branson/bransonlab/share/MCR/v91';


%% find data files
switch rawfiletype,
  case 'klb'
    if rawisregexp,
      inputdatafiles = mydir(rawdatadir,'name',rawfilestr,'recursive',rawisrecursive,'maxdepth',rawmaxdepth);
    else
      inputdatafiles = mydir(fullfile(rawdatadir,rawfilestr),'recursive',rawisrecursive,'maxdepth',rawmaxdepth);
    end
  case 'h5',
    inputdatafiles = mydir(fullfile(rawdatadir,'*h5'));
  otherwise,
    error('Not implemented');
end
if isempty(inputdatafiles),
  error('No input data files found');
end

% TODO: out of date, see rawfilestr
% needs review
% timestamp = regexp(inputdatafiles,'TM(\d+)_','once','tokens');
timestamp = regexp(inputdatafiles,'Volume_(\d+)','once','tokens');
timestamp = str2double([timestamp{:}]);
[timestamp,order] = sort(timestamp);
alltimestamp = timestamp;
inputdatafiles = inputdatafiles(order);
allinputdatafiles = inputdatafiles;
assert(all(timestamp == (0:numel(timestamp)-1)));

preddatafiles = mydir(fullfile(preddatadir,'*.h5'));
% TODO: out of date, needs review
m = regexp(preddatafiles,'/(\d+)\.h5','tokens','once');
m = [m{:}];
predtimestamps = str2double(m);
[predtimestamps,order] = sort(predtimestamps);
preddatafiles = preddatafiles(order);
[ism,idx] = ismember(predtimestamps,timestamp);
assert(all(ism));
inputdatafiles = inputdatafiles(idx);
timestamp = timestamp(idx);

%% save dense to sparse predictions

if ~exist(sparsepreddatadir,'dir'),
  mkdir(sparsepreddatadir);
end

% TODO: hard coded
SCRIPT = '/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/SaveSparsePredictions/for_redistribution_files_only/run_SaveSparsePredictions.sh';
ncores = 3;

for i = 1:numel(predtimestamps),

  [~,namecurr] = fileparts(preddatafiles{i});
  outmatfile = fullfile(sparsepreddatadir,[namecurr,'.mat']);
  if exist(outmatfile,'file'),
    continue;
  end

  if usecluster
  scriptfile = fullfile(sparsepreddatadir,['SaveSparse_',namecurr,'.sh']);
  logfile = fullfile(sparsepreddatadir,['SaveSparse_',namecurr,'.log']);

  jobid = sprintf('SaveSparse%d',i);
  fid = fopen(scriptfile,'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %s %d %f %d\n',...
    SCRIPT,MCR,preddatafiles{i},outmatfile,predtimestamps(i),predthresh,ncores);
  fclose(fid);

  unix(sprintf('chmod u+x %s',scriptfile));
  cmd = sprintf('ssh login1 ''source /etc/profile; bsub -n %d -J %s -o ''%s'' ''\"%s\"''''',...
    ncores,jobid,logfile,scriptfile);
  unix(cmd);

  else

    disp(i);
    SaveSparsePredictions(preddatafiles{i},outmatfile,predtimestamps(i),predthresh);

  end

end

% change names to new sparse versions
oldpreddatafiles = preddatafiles;
for i = 1:numel(predtimestamps),

  [~,namecurr] = fileparts(oldpreddatafiles{i});
  preddatafiles{i} = fullfile(sparsepreddatadir,[namecurr,'.mat']);
  assert(exist(preddatafiles{i},'file')>0);

end
predfiletype = 'mat';

%% load in all timepoints with predictions

timepoints_pa = predtimestamps;

i = 1;
[rawreadframe,rawnframes,rawfid,rawheaderinfo] = get_readframe_fcn(inputdatafiles{i});
rawsz = [rawheaderinfo.xyzct(1:3),max(predtimestamps)];
allpredidx = zeros(0,5);

for i = 1:numel(predtimestamps),

  fprintf('i = %d\n',i);

  switch predfiletype,
    case 'sparseh5',
      preddata = h5read(preddatafiles{i},preddatasetname);
      if isempty(preddata),
        continue;
      end
      preddata = preddata([3,2,1,4],:)';
      idxthresh = preddata(:,4) >= predthresh;
      fprintf('i = %d, nthresh = %d / %d\n',i,nnz(idxthresh),size(preddata,1));
      allpredidx(end+1:end+nnz(idxthresh),:) = [preddata(idxthresh,1:3),zeros(nnz(idxthresh),1)+predtimestamps(i),preddata(idxthresh,4)];
    case 'h5',
      preddata = h5read(preddatafiles{i},preddatasetname);
      if isempty(preddata),
        continue;
      end
      idxthresh = find(preddata>=predthresh);
      szcurr = size(preddata);
      assert(all(szcurr == rawsz(1:3)));
      locscurr = ind2subv(szcurr,idxthresh);
      scorescurr = preddata(idxthresh);
      fprintf('i = %d, nthresh = %d / %d\n',i,numel(idxthresh),numel(preddata));
      allpredidx(end+1:end+numel(idxthresh),:) = ...
        cat(2,locscurr,repmat(predtimestamps(i),[numel(idxthresh),1]),scorescurr);
    case 'mat',
      res = load(preddatafiles{i});
      allpredidx(end+1:end+size(res.allpredidx,1),:) = res.allpredidx;
    otherwise
      error('Not implemented');
  end

end

%% quality checks

T = max(predtimestamps);
npredpertimepoint = hist(allpredidx(:,4),1:T);
minv = nan(T,3);
maxv = nan(T,3);

for d = 1:3,
  minv(:,d) = accumarray(allpredidx(:,4),allpredidx(:,d),[T,1],@min,inf);
  maxv(:,d) = accumarray(allpredidx(:,4),allpredidx(:,d),[T,1],@max,-inf);
end

thresh_npred = 100000;
threshx = 1000;
threshy = 1000;
threshz = 100;

isoutlier_npred = npredpertimepoint <= thresh_npred;
if ~iscompletedata,
  isoutlier_npred(setdiff(1:T,predtimestamps)) = false;
end

dv = maxv - minv;
dv(isnan(dv)) = 0;
dv(isinf(dv)) = 0;
isoutlier_x = dv(:,1) <= threshx;
isoutlier_y = dv(:,2) <= threshy;
if ~iscompletedata,
  isoutlier_x(setdiff(1:T,predtimestamps)) = false;
  isoutlier_y(setdiff(1:T,predtimestamps)) = false;
end


if iscompletedata,
  b = regress(dv(:,3),[ones(1,T);1:T]');
else
  b = regress(dv(predtimestamps,3),[ones(1,numel(predtimestamps));predtimestamps(:)']');
end
dzfit = b(1) + (1:T).*b(2);
dfit = dzfit' - dv(:,3);
isoutlier_z = dfit >= threshz;
if ~iscompletedata,
  isoutlier_z(setdiff(1:T,predtimestamps)) = false;
end

hfig = 2;
figure(hfig);
clf;
hax = createsubplots(4,1,.05);
axes(hax(1));
plot(1:T,npredpertimepoint,'k.-');
hold on;
plot([1,T],[0,0] + thresh_npred,'r-');
plot(find(isoutlier_npred),npredpertimepoint(isoutlier_npred),'ro');
set(hax(1),'XLim',[0,T+1],'YLim',[-1,max(npredpertimepoint)+1]);
ylabel(sprintf('N. pred pixels >= %f',predthresh));
%set(hax(1),'YScale','log');

for d = 1:3,
  plot(hax(d+1),1:T,dv(:,d),'k.-');
%   hold(hax(d+1),'on');
%   plot(hax(d+1),1:T,maxv(:,d),'r.-');
  set(hax(d+1),'XLim',[0,T+1]);%,'YLim',[0,rawsz(d)+1]);
end

hold(hax(2),'on');
plot(hax(2),[1,T],[0,0]+threshx,'r-');
plot(hax(2),find(isoutlier_x),dv(isoutlier_x,1),'ro');

hold(hax(3),'on');
plot(hax(3),[1,T],[0,0]+threshy,'r-');
plot(hax(3),find(isoutlier_y),dv(isoutlier_y,2),'ro');

hold(hax(4),'on');
plot(hax(4),[1,T],dzfit([1,T])-threshz,'r-');
plot(hax(4),find(isoutlier_z),dv(isoutlier_z,3),'ro');


ylabel(hax(2),'x-span');
ylabel(hax(3),'y-span');
ylabel(hax(4),'z-span');
xlabel(hax(4),'Time');

isoutlier = isoutlier_x | isoutlier_y | isoutlier_z | isoutlier_npred';
for t = find(isoutlier'),

  fprintf('t = %d: npred = %d, x-span = %d, y-span = %d, z-span = %d\n',t,npredpertimepoint(t),dv(t,1),dv(t,2),dv(t,3));

end

%% rerun dense to sparse for bad data
timestampsrerun = find(isoutlier);
[~,idxrerun] = ismember(timestampsrerun,predtimestamps);
for ii = 1:numel(idxrerun),

  if idxrerun(ii) == 0,
    continue;
  end
  i = idxrerun(ii);

  [~,namecurr] = fileparts(oldpreddatafiles{i});
  tmp = dir(oldpreddatafiles{i});
  densedatadate = tmp.date;
  outmatfile = fullfile(sparsepreddatadir,[namecurr,'.mat']);
  if exist(outmatfile,'file'),
    res = load(outmatfile,'densedatadate');
    if isfield(res,'densedatadate') && res.densedatadate >= densedatadate,
      continue;
    end
  end

  if usecluster,
    scriptfile = fullfile(sparsepreddatadir,['SaveSparseRerun_',namecurr,'.sh']);
    logfile = fullfile(sparsepreddatadir,['SaveSparseRerun_',namecurr,'.log']);

    jobid = sprintf('SaveSparse%d',i);
    fid = fopen(scriptfile,'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %s %s %d %f %d\n',...
      SCRIPT,MCR,oldpreddatafiles{i},outmatfile,predtimestamps(i),predthresh,ncores);
    fclose(fid);

    unix(sprintf('chmod u+x %s',scriptfile));
    %   cmd = sprintf('ssh login1 ''source /etc/profile; qsub -pe batch %d -N %s -j y -b y -o ''%s'' -cwd ''\"%s\"''''',...
    %     ncores,jobid,logfile,scriptfile);
    cmd = sprintf('ssh login1 ''source /etc/profile; bsub -n %d -J %s -o ''%s'' ''\"%s\"''''',...
      ncores,jobid,logfile,scriptfile);
    unix(cmd);

  else

    SaveSparsePredictions(oldpreddatafiles{i},outmatfile,predtimestamps(i),predthresh);

  end

end

%% plot max projections for some sample timepoints

for i = 1:20:numel(predtimestamps),

  t = predtimestamps(i);
  %idx = allpredidx(:,4)==predtimestamps(i);
  predmaxv = ConstructPredMaxProj2(allpredidx,rawsz(1:3),predtimestamps(i));
  figure(i);
  clf;
  imagesc(predmaxv); axis image;
  truesize;
  hold on;

  savefig_pa(fullfile(savedir,sprintf('MaxProjection_t%03d.png',t)),i,'png');

end

%% filtering & non-maximal suppression

if ~usecluster,
  error('This part of the code is written to use the cluster only');
end

% nonmaximal suppression
sz = [rawsz(1:3),max(allpredidx(:,4))];
filsig = [5,5,5,2];
filrad = filsig;
ncores = 2;

nchunksperjob = 100;

nchunksperjob_conncomp = 3000;

chunksize = [100,100,50,20];
chunkstep = max(1,floor(chunksize/2));

chunksize_conncomp = [100,100,50,20];
chunkstep_conncomp = max(1,floor(chunksize_conncomp/2));


nsplit = 10;
splitstarts = round(linspace(1,T,nsplit));
splitends = [splitstarts(2:end)-1,T];
splitstarts_buffer = splitstarts - filrad(4) - 2;
splitends_buffer = splitends + filrad(4) + 2;

inmatfiles = cell(1,nsplit);
for spliti = 1:nsplit,

  i0 = splitstarts_buffer(spliti);
  i1 = splitends_buffer(spliti);
  j0 = find(allpredidx(:,4)>=i0,1);
  j1 = find(allpredidx(:,4)<=i1,1,'last');

  id = sprintf('%d_%dto%d',spliti,i0,i1);
  inmatfile = sprintf('%s_%s.mat',inmatfilestr,id);
  outmatfilestr = ['RegionalMax',id];
  scriptfilestr = ['RegionalMax',id];
  logfilestr = ['RegionalMax',id];

  inmatfiles{spliti} = inmatfile;

  imregionalmax_list2(allpredidx(j0:j1,:),sz,'filsig',filsig,'filrad',filrad,'chunksize',chunksize,'chunkstep',chunkstep,...
    'fixrunoncluster',true,'outdir',savedir,'inmatfile',inmatfile,'ncores',ncores,'nchunksperjob',nchunksperjob,...
    'scriptfilestr',scriptfilestr,'outmatfilestr',outmatfilestr,'logfilestr',logfilestr);

end

nonmaxpredidx0 = cell(1,nsplit);
isdone = cell(1,nsplit);

for spliti = 1:nsplit,

  i0 = splitstarts_buffer(spliti);
  i1 = splitends_buffer(spliti);
  j0 = find(allpredidx(:,4)>=i0,1);
  j1 = find(allpredidx(:,4)<=i1,1,'last');

  inmatfile = inmatfiles{spliti};

  [nonmaxpredidx0{spliti},isdone{spliti}] = imregionalmax_list2([],[],'finishrunoncluster',true,'inmatfile',inmatfile);

end

nonmaxpredidx = cell(1,nsplit);
inmatfiles_conncomp = cell(1,nsplit);
for spliti = 1:nsplit,

  i0 = splitstarts_buffer(spliti);
  i1 = splitends_buffer(spliti);
  j0 = find(allpredidx(:,4)>=i0,1);
  j1 = find(allpredidx(:,4)<=i1,1,'last');

  minidx = [1,1,1,i0];
  maxidx = [sz(1:3),i1];

  id = sprintf('%d_%dto%d',spliti,i0,i1);
  inmatfile = sprintf('%s_%s.mat',inmatfilestr_conncomp,id);
  outmatfilestr = ['ConnComp',id];
  scriptfilestr = ['ConnComp',id];
  logfilestr = ['ConnComp',id];

  inmatfiles_conncomp{spliti} = inmatfile;

  bwconncomp_list2(nonmaxpredidx0{spliti},sz,'chunksize',chunksize_conncomp,'chunkstep',chunkstep_conncomp,...
    'minidx',minidx,'maxidx',maxidx,'outdir',savedir,...
    'startrunoncluster',true,'inmatfile',inmatfile,'ncores',ncores,'nchunksperjob',nchunksperjob_conncomp,...
    'scriptfilestr',scriptfilestr,'outmatfilestr',outmatfilestr,'logfilestr',logfilestr);

end

isdone2 = cell(1,nsplit);
for spliti = 1:nsplit,

  i0 = splitstarts_buffer(spliti);
  i1 = splitends_buffer(spliti);
  j0 = find(allpredidx(:,4)>=i0,1);
  j1 = find(allpredidx(:,4)<=i1,1,'last');

  inmatfile = inmatfiles_conncomp{spliti};

  [nonmaxpredidx{spliti},isdone2{spliti}] = bwconncomp_list2([],[],'finishrunoncluster',true,'inmatfile',inmatfile);

end

% which split should we grab from
bestdend = -inf(1,T);
splitis = nan(1,T);
for spliti = 1:nsplit,

  i0 = splitstarts_buffer(spliti);
  i1 = splitends_buffer(spliti);
  tscurr = max(i0,1):min(i1,T);

  dend = min(tscurr - i0,i1-tscurr);
  idxbest = find(dend > bestdend(tscurr));
  bestdend(tscurr(1)+idxbest-1) = dend(idxbest);
  splitis(tscurr(1)+idxbest-1) = spliti;

end

nonmaxpredidx_combined = zeros(0,5);
for spliti = 1:nsplit,
  ts = find(splitis==spliti);
  idxcurr = ismember(round(nonmaxpredidx{spliti}(:,4)),ts);
  ncurr = nnz(idxcurr);
  nonmaxpredidx_combined(end+1:end+ncurr,:) = nonmaxpredidx{spliti}(idxcurr,:);
end

scores = nonmaxpredidx_combined(:,end);
peaks = nonmaxpredidx_combined(:,1:4);

save(fullfile(savedir,'detections.mat'),'nonmaxpredidx0','nonmaxpredidx','nonmaxpredidx_combined','scores','peaks');

% per-timepoint

% nonmaxpredidx0_pert = {};
% nonmaxpredidx_pert = {};
% scores_pert = {};
% peaks_pert = {};
%
% for ti = 1:numel(predtimestamps),
%   t = predtimestamps(ti);
%
%   chunksize = [100,100,50,2];
%   chunkstep = chunksize/2;
%
%   idxcurr = allpredidx(:,4)==t;
%   newallpredidxcurr = imregionalmax_list(allpredidx(idxcurr,:),[sz(1:3),t],'filrad',filrad,'chunksize',chunksize,'chunkstep',chunkstep);
%   nonmaxpredidx0_pert{ti} = newallpredidxcurr;
%
%   chunksize = [100,100,50,4];
%   chunkstep = chunksize/2;
%
%   newallpredidx2curr = bwconncomp_list(newallpredidxcurr,[sz(1:3),t],'chunksize',chunksize,'chunkstep',chunkstep);
%   nonmaxpredidx_pert{ti} = newallpredidx2curr;
%
%   scores_pert{ti} = newallpredidx2curr(:,end);
%   peaks_pert{ti} = newallpredidx2curr(:,1:4);
%
%   save detections4.mat nonmaxpredidx0_pert nonmaxpredidx_pert scores_pert peaks_pert ti
% end

%% plot detections

if ~exist('labellocs','var'),
  labellocs = zeros(0,4);
end
ti = 2;
thresh = .001;
boxrad = [100,100,5,3];
figpos = [10,10,2400,760];
nplot = 5;
ntimestampsplot = 5;
timestampsplot = randsample(predtimestamps,ntimestampsplot);

for ti = 1:ntimestampsplot,

  t = timestampsplot(ti);

  %t = predtimestamps(ti);
  idxcurr = find(round(peaks(:,4)) == t);
  scorescurr = scores(idxcurr);
  peakscurr = peaks(idxcurr,:);

  %scorescurr = scores_pert{ti};
  %peakscurr = peaks_pert{ti};

  [sortedscores,order] = sort(scorescurr,1,'descend');

  for detorderi = 1:min(nplot,numel(sortedscores)),

    deti = order(detorderi);
    loc = peakscurr(deti,:);

    hfig = 10+detorderi;

    textstr0 = sprintf(' Det rank %d, score = %.1f',detorderi,scores(deti));

    filename = fullfile(savedir,sprintf('Rank%ddet%d_score%.1f_t%dx%dy%dz%d.png',...
      detorderi,deti,scores(deti),round(loc(4)),round(loc(1)),round(loc(2)),round(loc(3))));

    PlotDivision(allinputdatafiles,alltimestamp,allpredidx,loc,rawsz,...
      'allscores',scores,'allpeaks',peaks,...%'scores_pert',scores_pert,'peaks_pert',peaks_pert,'timepoints_pa',predtimestamps,...
      'labellocs',labellocs,...
      'thresh',thresh,...
      'boxrad',boxrad,...
      'hfig',hfig,...
      'textstr0',textstr0,...
      'figpos',figpos,'filename',filename);

    drawnow;

  end

end

%% plot labels

thresh = .001;
boxrad = [100,100,5,3];
figpos = [10,10,2400,760];

labelidx = find(ismember(labellocs(:,4),predtimestamps));

for labelii = 1:numel(labelidx),

  labeli = labelidx(labelii);
  loc = labellocs(labeli,:);
  t = loc(4);

  hfig = 20;

  filename = fullfile(savedir,sprintf('Label%d_t%dx%dy%dz%d.png',...
    labeli,round(loc(3)),round(loc(1)),round(loc(2)),round(loc(3))));

  PlotDivision(allinputdatafiles,alltimestamp,allpredidx,loc,rawsz,...
    'allscores',scores,'allpeaks',peaks,...%'scores_pert',scores_pert,'peaks_pert',peaks_pert,'timepoints_pa',predtimestamps,...
    'labellocs',labellocs,...
    'thresh',thresh,...
    'boxrad',boxrad,...
    'hfig',hfig,...
    'figpos',figpos,'filename',filename);

  drawnow;

end

%% output table of results

fid = fopen(fullfile(savedir,'Threshold2PrecisionRecall.csv'),'w');
fprintf(fid,'Threshold,TestPrecision,TestRecall\n');
for i = 1:numel(thresholds_try),
  fprintf(fid,'%e,%e,%e\n',thresholds_try(i),precision(i),recall(i));
end
fclose(fid);

fid = fopen(fullfile(savedir,'Predictions.csv'),'w');
fprintf(fid,'X,Y,Z,T,Score\n');
for i = 1:size(peaks,1),
  fprintf(fid,'%f,%f,%f,%f,%e\n',peaks(i,1),...
    peaks(i,2),peaks(i,3),peaks(i,4),scores(i));
end

% for ti = 1:numel(peaks_pert),
%   t = predtimestamps(ti);
%   for i = 1:size(peaks_pert{ti},1),
%     fprintf(fid,'%f,%f,%f,%f,%e\n',peaks_pert{ti}(i,1),...
%       peaks_pert{ti}(i,2),peaks_pert{ti}(i,3),t,scores_pert{ti}(i));
%   end
% end
fclose(fid);

%% plot stuff

maxprojfiles = mydir(fullfile(maxprojdatadir,'*yzProjection*.klb'));
m = regexp(maxprojfiles,'_TM(\d+)_','once','tokens');
maxprojtimestamps = str2double([m{:}]);
[maxprojtimestamps,order] = sort(maxprojtimestamps);
maxprojfiles = maxprojfiles(order);

%%

cdd = struct;
%cdd.filsig = [50,50,10,4];
cdd.filsig = [12.5,12.5,2.5,4];
cdd.filrad = ceil(2.5*cdd.filsig);
cdd.filrad(4) = 4;
cdd.thresh = 0.084;
cdd.ncores = 4;
cdd.list = nonmaxpredidx_combined;
cdd.sz = [rawsz(1:3),max(allpredidx(:,4))];
cdd.stepsz = [220,220,100,9];
cdd.dim = 1;
% TODO: hard coded
sparsepreddatadir = '/nrs/branson/MouseLineaging/ComputeDivisionDensity_yz_v2';
inmatfile = '/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/ComputeDivisionDensityInput_yz_v2.mat';
save(inmatfile,'-struct','cdd');

if ~exist(sparsepreddatadir,'dir'),
  mkdir(sparsepreddatadir);
end

% TODO: hard coded
SCRIPT = '/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/ComputeDivisionDensity/for_redistribution_files_only/run_ComputeDivisionDensity.sh';

for t = 1:T

  scriptfile = fullfile(sparsepreddatadir,sprintf('%d.sh',t));
  logfile = fullfile(sparsepreddatadir,sprintf('%d.log',t));
  outmatfile = fullfile(sparsepreddatadir,sprintf('%d.mat',t));
  if exist(outmatfile,'file'),
    continue;
  end
  jobid = sprintf('CDD%d',t);
  fid = fopen(scriptfile,'w');
  fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
  fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
  fprintf(fid,'fi\n');
  fprintf(fid,'%s %s %s %d %s\n',...
    SCRIPT,MCR,inmatfile,t,outmatfile);
  fclose(fid);

  unix(sprintf('chmod u+x %s',scriptfile));
  cmd = sprintf('ssh login1 ''source /etc/profile; bsub -n %d -J %s -o ''%s'' ''\"%s\"''''',...
    cdd.ncores,jobid,logfile,scriptfile);
  unix(cmd);

end


maxint = 1000;
maxrho = 1.25e-06;
maxrhos = nan(1,T);
maxints = nan(1,T);
for t = 1:T,
  if isnan(maxints(t)),
    i = find(maxprojtimestamps == t);
    maxprojim = readKLBstack(maxprojfiles{i});
    maxints(t) = prctile(maxprojim(maxprojim>0),99.9);
  end
  if isnan(maxrhos(t)),
    outmatfile = fullfile(sparsepreddatadir,sprintf('%d.mat',t));
    if ~exist(outmatfile,'file'),
      continue;
    end
    tmp = load(outmatfile);
    maxrhos(t) = max(tmp.maxprojdiv(:));
  end
end

fil = normpdf(-50:50,0,20);
maxintsmooth = imfilter(maxints,fil,'same','symmetric');
maxrho = nanmedian(maxrhos);
%
% coeffsmaxint = regress(maxints',[1:T;ones(1,T)]');
% maxintsfit = coeffsmaxint(1).*(1:T) + coeffsmaxint(2);

vidobj = VideoWriter('Divisions_yz_v2.avi');
vidobj.FrameRate = 10;
open(vidobj);

otherdims = setdiff(1:3,cdd.dim);

hfig = 1;
figure(hfig);
clf;
hax = gca;
for t = 1:T,

  outmatfile = fullfile(sparsepreddatadir,sprintf('%d.mat',t));
  if ~exist(outmatfile,'file'),
    break;
  end
  tmp = load(outmatfile);
  i = find(maxprojtimestamps == t);
  maxprojim = readKLBstack(maxprojfiles{i});
%   i = find(alltimestamp==t);
%   maxprojim = zeros(sz(1:2));
%   uniquezs = unique(tmp.zdiv(:));
%   for z = uniquezs(:)',
%     if z == 0,
%       continue;
%     end
%     slice = readKLBslice(allinputdatafiles{i},z,3);
%     idxcurr = tmp.zdiv == z;
%     maxprojim(idxcurr) = slice(idxcurr);
%   end


  [~,~,colorim] = PlotDivisionDensity(maxprojim,tmp.maxprojdiv,'hax',hax,'maxint',maxintsmooth(t)*.75,'maxrho',maxrho);
  title(num2str(t));
  drawnow;
  if t == 1,
    newsz = round(size(colorim).*[isoscale(otherdims),1]/4)*4;
    newsz = newsz(1:2);
  end
  colorim = max(0,min(1,imresize(colorim,newsz)));
%   image(colorim,'Parent',hax);
%   truesize;
%   axis(hax,'image');
%   truesize;
%   drawnow;
  writeVideo(vidobj,colorim);
end
close(vidobj);

%% parameter sweep

filsigs_try = ...
  [50,50,10,4
  25,25,5,4
  10,10,2,4
  40,40,8,4
  30,30,6,4
  20,20,4,4
  15,15,3,4];

inmatfiles = cell(1,size(filsigs_try,1));
for i = 1:size(filsigs_try,1),

  inmatfiles{i} = sprintf('/groups/branson/home/bransonk/tracking/code/Lineaging/MouseEmbryo/ComputeDivisionDensityInput_xz_%d.mat',i);
  if exist(inmatfiles{i},'file'),
    continue;
  end
  cddcurr = cdd;
  cddcurr.filsig = filsigs_try(i,:);
  cddcurr.filrad(1:3) = ceil(cddcurr.filsig(1:3).*2.5);
  save(inmatfiles{i},'-struct','cddcurr');

end

tstry = [66, 189, 293, 377];
% TODO: hard coded
sparsepreddatadir = '/nrs/branson/MouseLineaging/ComputeDivisionDensity_yz_paramsweep';
if ~exist(sparsepreddatadir,'dir'),
  mkdir(sparsepreddatadir);
end

for i = 1:size(filsigs_try,1),

  for ti = 1:numel(tstry),

    t = tstry(ti);
    scriptfile = fullfile(sparsepreddatadir,sprintf('%d_%d.sh',t,i));
    logfile = fullfile(sparsepreddatadir,sprintf('%d_%d.log',t,i));
    outmatfile = fullfile(sparsepreddatadir,sprintf('%d_%d.mat',t,i));
    if exist(outmatfile,'file'),
      continue;
    end
    jobid = sprintf('CDD%d_%d',t,i);
    fid = fopen(scriptfile,'w');
    fprintf(fid,'if [ -d %s ]\n',TMP_ROOT_DIR);
    fprintf(fid,'  then export MCR_CACHE_ROOT=%s.%s\n',MCR_CACHE_ROOT,jobid);
    fprintf(fid,'fi\n');
    fprintf(fid,'%s %s %s %d %s\n',...
      SCRIPT,MCR,inmatfiles{i},t,outmatfile);
    fclose(fid);

    unix(sprintf('chmod u+x %s',scriptfile));
    cmd = sprintf('ssh login1 ''source /etc/profile; bsub -n %d -J %s -o ''%s'' ''\"%s\"''''',...
      cdd.ncores,jobid,logfile,scriptfile);
    unix(cmd);

  end
end

hfig = 2;
figure(hfig);
clf;
hax = createsubplots(numel(tstry),size(filsigs_try,1),.01);
hax = reshape(hax,[numel(tstry),size(filsigs_try,1)]);

[~,filorder] = sort(filsigs_try(:,1));

for ti = 1:numel(tstry),
  t = tstry(ti);
  i = find(maxprojtimestamps == t);
  maxprojim = readKLBstack(maxprojfiles{i});
  for filii = 1:size(filsigs_try,1),

    fili = filorder(filii);
    outmatfile = fullfile(sparsepreddatadir,sprintf('%d_%d.mat',t,fili));
    if ~exist(outmatfile,'file'),
      continue;
    end
    tmp = load(outmatfile);

    [~,~,colorim] = PlotDivisionDensity(maxprojim,tmp.maxprojdiv,'hax',hax(ti,filii),'maxint',maxintsmooth(t)*.75,'maxrho',max(tmp.maxprojdiv(:)));
    newsz = [size(colorim,1),size(colorim,2)].*isoscale(otherdims);
    colorim = max(0,min(1,imresize(colorim,newsz)));
    image(colorim,'Parent',hax(ti,filii));
    axis(hax(ti,filii),'image','off');
    drawnow;
  end
  linkaxes(hax);
end


%% OLD CODE

% %% compute accuracy at different thresholds
%
% test_timepoints = [120,240,360];
% labeldatafiles = [2,3];
% isoscale = [1,1,5];
%
% istestlabel = ismember(labellocs(:,4),test_timepoints) & ismember(datafilei,labeldatafiles);
% testpredidx = find(ismember(predtimestamps,test_timepoints));
%
% thresholds_try = linspace(10^-5,.5,100);
% %thresholds_try = logspace(-5,log10(.5),1000);
% match_rad = [50,2];
%
% hfig = 122;
% figure(hfig);
% clf(hfig);
%
% [precision,recall,hfig,hax] = ...
%   ComputePrecisionRecall(labellocs(istestlabel,:),peaks,scores,thresholds_try,test_timepoints,...
%   'match_rad',match_rad,'isoscale',isoscale,'hfig',hfig,'doplot',true,'plotcolor','k');
%
% % scores_cat = cat(1,scores_pert{testpredidx});
% % peaks_cat = cat(1,peaks_pert{testpredidx});
% % labellocs_cat = labellocs(istestlabel,:);
% %
% % [precision,recall,ntruepos,nfalsepos,nfalseneg,hfig,hax] = ...
% %   MatchLabelsAndPredictions(labellocs_cat,peaks_cat,scores_cat,thresholds_try,...
% %   'match_rad',match_rad,'hfig',hfig,'doplot',true,'plotcolor','k');

% %% train timepoints

% timepoints_pred = predtimestamps(~cellfun(@isempty,scores_pert));
% train_timepoints = setdiff(timepoints_pred,test_timepoints);
%
% istrainlabel = ismember(labellocs(:,4),train_timepoints) & ismember(datafilei,labeldatafiles);
% trainpredidx = find(ismember(predtimestamps,train_timepoints));
%
% scores_cat = cat(1,scores_pert{trainpredidx});
% peaks_cat = cat(1,peaks_pert{trainpredidx});
% labellocs_cat = labellocs(istrainlabel,:);
%
% [precision_train,recall_train,ntruepos_train,nfalsepos_train,nfalseneg_train,hfig,hax] = ...
%   MatchLabelsAndPredictions(labellocs_cat,peaks_cat,scores_cat,thresholds_try,...
%   'match_rad',match_rad,'hfig',hfig,'hax',hax,'doplot',true,'plotcolor','r');

% %% precision-recall for tgmm
%
% load(tgmmfile);
%
%
% [precision_tgmm,recall_tgmm] = ...
%   ComputePrecisionRecall(labellocs(istestlabel,:),tgmmDivisions,ones(size(tgmmDivisions,1),1),.5,test_timepoints,...
%   'match_rad',match_rad,'isoscale',isoscale,'doplot',false);
% %
% % idxtest = ismember(tgmmDivisions(:,4),test_timepoints);
% % npredtgmm = nnz(idxtest);
% % istestlabel = ismember(labellocs(:,4),test_timepoints) & ismember(datafilei,labeldatafiles);
% % labellocs_cat = labellocs(istestlabel,:);
% %
% % [precision_tgmm,recall_tgmm,ntruepos_tgmm,nfalsepos_tgmm,nfalseneg_tgmm] = ...
% %   MatchLabelsAndPredictions(labellocs_cat,tgmmDivisions(idxtest,:),ones(nnz(idxtest),1),.5,...
% %   'match_rad',match_rad,'doplot',false);

% %% results per timepoint
%
% colors = jet(numel(test_timepoints))*.7;
%
% precision_pert = nan(numel(thresholds_try),numel(test_timepoints));
% recall_pert = nan(numel(thresholds_try),numel(test_timepoints));
%
% for i = 1:numel(test_timepoints),
%
%   t = test_timepoints(i);
%
%   islabel = (labellocs(:,4)==t) & ismember(datafilei,labeldatafiles);
%
%   [precision_pert(:,i),recall_pert(:,i),hfig,hax] = ...
%     ComputePrecisionRecall(labellocs(islabel,:),peaks,scores,thresholds_try,t,...
%     'match_rad',match_rad,'isoscale',isoscale,'hfig',hfig,'hax',hax,'doplot',true,'plotcolor',colors(i,:));
%
% %   [precision_pert(:,i),recall_pert(:,i),ntruepos_pert(:,i),nfalsepos_pert(:,i),nfalseneg_pert(:,i),hfig,hax] = ...
% %     MatchLabelsAndPredictions(labellocs(islabel,:),peaks_pert{i},scores_pert{i},thresholds_try,...
% %     'match_rad',match_rad,'hfig',hfig,'hax',hax,'doplot',true,'plotcolor',colors(i,:));
% end


% %% plot results
%
% hfig = 123;
% figure(hfig);
% clf;
% h = [];
% legs = {};
% legs{end+1} = sprintf('CNN, test timepoints %s',mat2str(test_timepoints));
% h(end+1) = plot(recall,precision,'ko-','MarkerFaceColor','k','LineWidth',2);
% hold on;
% % h(end+1) = plot(recall_train,precision_train,'ro-','MarkerFaceColor','r','LineWidth',2);
% % legs{end+1} = sprintf('CNN, train timepoints %s',mat2str(train_timepoints));
% h(end+1) = plot(recall_tgmm,precision_tgmm,'co','MarkerFaceColor','c','LineWidth',2);
% legs{end+1} = sprintf('TGMM, test timepoints %s',mat2str(test_timepoints));
% trainmarker = '.';
% trainlinestyles = {':','--'};
% testmarker = '.';
% testlinestyle = '-';
%
% for i = 1:numel(test_timepoints),
%   legs{end+1} = sprintf('CNN, timepoint %d',test_timepoints(i));
%   marker = [testmarker,testlinestyle];
%   h(end+1) = plot(recall_pert(:,i),precision_pert(:,i),marker,'Color',colors(i,:));
% end
%
% xlabel('Recall');
% ylabel('Precision');
% legend(h,legs);
% box off;
% set(gca,'XLim',[-.01,1.01],'YLim',[-.01,1.01]);
%
%
%
% %% include new labels
%
% newlabelfiles = mydir('.','name','Evaluation T.* Kate Final.csv');
% m = regexp(newlabelfiles,'T(\d+) ','once','tokens');
% newlabelfiletimepoints = str2double([m{:}]);
% labelspos = [1];
%
% newlabellocs = zeros(0,4);
% for i = 1:numel(newlabelfiles),
%
%   fid = fopen(newlabelfiles{i},'r');
%   s = fgetl(fid);
%   ss = regexp(s,',','split');
%   xi = find(strcmp(ss,'X'));
%   yi = find(strcmp(ss,'Y'));
%   zi = find(strcmp(ss,'Z'));
%   ti = find(strcmp(ss,'T'));
%   li = find(strcmp(ss,'Label'));
%
%   while true,
%     s = fgetl(fid);
%     if ~ischar(s),
%       break;
%     end
%     s = strtrim(s);
%     if isempty(s),
%       continue;
%     end
%     ss = regexp(s,',','split');
%     label = str2double(ss{li});
%     if ~ismember(label,labelspos),
%       continue;
%     end
%     loc = str2double(ss([xi,yi,zi,ti]));
%     newlabellocs(end+1,:) = loc;
%   end
%   fclose(fid);
%
% end
%
%
% %% compute accuracy at different thresholds
%
% test_timepoints = [120,240,360];
% labeldatafiles = [2,3];
% isoscale = [1,1,5];
%
% istestlabel = ismember(labellocs(:,4),test_timepoints) & ismember(datafilei,labeldatafiles);
% isnewtestlabel = ismember(newlabellocs(:,4),test_timepoints);
% testpredidx = find(ismember(predtimestamps,test_timepoints));
%
% thresholds_try = linspace(10^-5,.5,100);
% %thresholds_try = logspace(-5,log10(.5),1000);
% match_rad = [50,2];
%
% hfig = 124;
% figure(hfig);
% clf(hfig);
%
% [newprecision,newrecall,hfig,hax] = ...
%   ComputePrecisionRecall([labellocs(istestlabel,:);newlabellocs(isnewtestlabel,:)],peaks,scores,thresholds_try,test_timepoints,...
%   'match_rad',match_rad,'isoscale',isoscale,'hfig',hfig,'doplot',true,'plotcolor','k');
%
% %% tgmm
%
% [newprecision_tgmm,newrecall_tgmm] = ...
%   ComputePrecisionRecall([labellocs(istestlabel,:);newlabellocs(isnewtestlabel,:)],tgmmDivisions,ones(size(tgmmDivisions,1),1),.5,test_timepoints,...
%   'match_rad',match_rad,'isoscale',isoscale,'doplot',false);
% %
% % idxtest = ismember(tgmmDivisions(:,4),test_timepoints);
% % npredtgmm = nnz(idxtest);
% % istestlabel = ismember(labellocs(:,4),test_timepoints) & ismember(datafilei,labeldatafiles);
% % labellocs_cat = labellocs(istestlabel,:);
% %
% % [precision_tgmm,recall_tgmm,ntruepos_tgmm,nfalsepos_tgmm,nfalseneg_tgmm] = ...
% %   MatchLabelsAndPredictions(labellocs_cat,tgmmDivisions(idxtest,:),ones(nnz(idxtest),1),.5,...
% %   'match_rad',match_rad,'doplot',false);
%
% %% human
%
% [newprecision_human,newrecall_human] = ...
%   ComputePrecisionRecall([labellocs(istestlabel,:);newlabellocs(isnewtestlabel,:)],labellocs(istestlabel,:),ones(nnz(istestlabel),1),.5,test_timepoints,...
%   'match_rad',match_rad,'isoscale',isoscale,'doplot',false);
%
%
% %% results per timepoint
%
% colors = jet(numel(test_timepoints))*.7;
%
% newprecision_pert = nan(numel(thresholds_try),numel(test_timepoints));
% newrecall_pert = nan(numel(thresholds_try),numel(test_timepoints));
%
% for i = 1:numel(test_timepoints),
%
%   t = test_timepoints(i);
%
%   islabel = (labellocs(:,4)==t) & ismember(datafilei,labeldatafiles);
%   isnewlabel = newlabellocs(:,4)==t;
%
%   [newprecision_pert(:,i),newrecall_pert(:,i),hfig,hax] = ...
%     ComputePrecisionRecall([labellocs(islabel,:);newlabellocs(isnewlabel,:)],peaks,scores,thresholds_try,t,...
%     'match_rad',match_rad,'isoscale',isoscale,'hfig',hfig,'hax',hax,'doplot',true,'plotcolor',colors(i,:));
%
% %   [precision_pert(:,i),recall_pert(:,i),ntruepos_pert(:,i),nfalsepos_pert(:,i),nfalseneg_pert(:,i),hfig,hax] = ...
% %     MatchLabelsAndPredictions(labellocs(islabel,:),peaks_pert{i},scores_pert{i},thresholds_try,...
% %     'match_rad',match_rad,'hfig',hfig,'hax',hax,'doplot',true,'plotcolor',colors(i,:));
% end
%
%
% %% plot results
%
% hfig = 125;
% figure(hfig);
% clf;
% h = [];
% legs = {};
% legs{end+1} = sprintf('CNN, test timepoints %s',mat2str(test_timepoints));
% h(end+1) = plot(newrecall,newprecision,'ko-','MarkerFaceColor','k','LineWidth',2);
% hold on;
% % h(end+1) = plot(recall_train,precision_train,'ro-','MarkerFaceColor','r','LineWidth',2);
% % legs{end+1} = sprintf('CNN, train timepoints %s',mat2str(train_timepoints));
% tgmmcolor = [.6,.6,.6];
% h(end+1) = plot(newrecall_tgmm,newprecision_tgmm,'o','Color',tgmmcolor,'MarkerFaceColor',tgmmcolor,'LineWidth',2);
% legs{end+1} = sprintf('TGMM, test timepoints %s',mat2str(test_timepoints));
% humancolor = [.8,.4,0];
% h(end+1) = plot(newrecall_human,newprecision_human,'o','Color',humancolor,'MarkerFaceColor',humancolor,'LineWidth',2);
% legs{end+1} = sprintf('Human, test timepoints %s',mat2str(test_timepoints));
%
% trainmarker = '.';
% trainlinestyles = {':','--'};
% testmarker = '.';
% testlinestyle = '-';
%
% for i = 1:numel(test_timepoints),
%   legs{end+1} = sprintf('CNN, timepoint %d',test_timepoints(i));
%   marker = [testmarker,testlinestyle];
%   h(end+1) = plot(newrecall_pert(:,i),newprecision_pert(:,i),marker,'Color',colors(i,:));
% end
%
% xlabel('Recall');
% ylabel('Precision');
% legend(h,legs);
% box off;
% set(gca,'XLim',[-.01,1.01],'YLim',[-.01,1.01]);
%
% %%
%
% save(fullfile(savedir,'mk5_large_small_batch_balanced_good_reweight.mat'));