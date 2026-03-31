LiteratureReviewSummary
## Thefollowinganalysisexamineselevenseminalandrecentresearchpaperstosynthesizethe
evolutionandcurrentstateofdistributedSGDarchitectures.
1.ScalingDistributedMachineLearningwiththeParameterServer(Lietal.,
## 2014)
Lietal.proposedathird-generationParameterServer(PS)frameworkdesignedtohandlethe
scaleofindustrialmachinelearningproblemsinvolvingupto10^{12}parameters.Thecore
innovationofthisworkwasthetreatmentofparametersasagloballysharedkey-valuestore,
partitionedacrossagroupofservernodestoeliminatesingle-pointbottlenecks.Thesystem
introducedrange-basedvectorclockstotrackparameterversionsacrossadistributedgraph,
allowingforflexibleconsistencymodelslikeSSPandASP.
●Contribution:Demonstratedelasticscalability,whereserversandworkerscanbe
addedorremovedwithoutrestartingthejob,andfaulttolerancethroughlivereplication.
●TechnicalDetail:Useda"KKTfilter"tosuppressthetransmissionofinsignificant
gradients,reducingnetworktrafficbyover93%insparsemodeltraining.
2.Horovod:FastandEasyDistributedDeepLearninginTensorFlow
(Sergeev&DelBalso,2018)
InresponsetothecomplexityandscalingissuesofearlyTensorFlowPSimplementations,
SergeevandDelBalsointroducedHorovod.ByutilizingtheRingAll-Reducealgorithm,
Horovodallowedfornear-linearscalingoftrainingthroughputonhigh-performanceclusters.
●Contribution:Introduced"TensorFusion,"anoptimizationthatbatchessmallgradients
intoasinglelargebuffertominimizetheoverheadofmultipleAll-Reducecalls.
●TechnicalDetail:LeveragedMPI(MessagePassingInterface)andNVIDIA’sNCCLfor
highlyoptimizedGPU-to-GPUcommunication,enablingefficientuseofhigh-speed
interconnectslikeNVLink.
3.Accurate,LargeMinibatchSGD:TrainingImageNetin1Hour(Goyalet
al.,2017)
ThisworkfocusedonthestatisticalchallengesofscalingsynchronousSGD.Bytraininga
ResNet-50modelon256GPUswithamassivebatchsizeof8,192,theauthorsprovedthat
synchronousRARcouldmatchtheaccuracyofsmall-batchtrainingifhyperparameterswere
scaledcorrectly.
●Contribution:Establishedthe"LinearScalingRule"(multiplyinglearningratebykwhen
batchsizeismultipliedbyk)andagradualwarmupstrategytostabilizeearlytraining.
●TechnicalDetail:Demonstrated90%scalingefficiencywhenmovingfrom8to256
GPUsoncommodityhardware.

4.CanDecentralizedAlgorithmsOutperformCentralizedAlgorithms?(Lian
etal.,2017)
Lianetal.providedarigoroustheoreticalfoundationfordecentralizedSGD(D-SGD),proving
thatitcanachievethesameasymptoticlinearspeedupascentralizedSGDwhileavoidingthe
server-sidebandwidthbottleneck.
●Contribution:Showedthatinbandwidth-limitedenvironments,D-SGDoutperformsthe
PS-workermodelbecausecommunicationisdistributedevenlyacrossallnodes.
●TechnicalDetail:HighlightedthattheeffectivenessofD-SGDdependsonthespectral
gapofthecommunicationgraph;well-connectedgraphs(likeexponentialgraphs)
generalizebetterthansparserings.
5.DeepLearningwithElasticAveragingSGD(Zhangetal.,2015)
TheEASGDalgorithmintroducedanovel"elasticforce"mechanismtolinklocalworker
parameterswithacentralvariableontheserver.UnlikestandardPSmodels,workersarenot
requiredtopullthefullmodelateverystep;theycommunicateperiodicallytopullthecenter
variabletowardtheirlocalstate.
●Contribution:Enabledmoreexplorationbyallowinglocalvariablestofluctuatefurther
fromthecenter,whichhelpsinescapinglocaloptimainnon-convexlandscapes.
●TechnicalDetail:Foundthatallowingasynchronyandmodel"slack"canimprove
convergencespeedandreducecommunicationfrequencycomparedtosimpleperiodic
averaging.
6.BytePS:AHigh-PerformanceHeterogeneousParameterServer(Jianget
al.,2020)
BytePSre-envisionedthePSarchitectureformoderncloudclusterswheremachineshave
multipleGPUsandsignificantCPUresources.ItutilizesbothGPUsandCPUsforaggregation,
employingahierarchicalstrategytooptimizecommunication.
●Contribution:Achievedupto2timesthetrainingspeedofHorovod+NCCLin
BERT-largetrainingbyleveragingunderutilizedCPUbandwidthforsummation.
●TechnicalDetail:Re-implementedinter-machinecommunicationtoavoidMPI
overhead,usingpriority-basedschedulingandtensorpartitioningtooverlap
communicationwithcomputation.
7.StragglAR:AcceleratingAll-ReducewithaPersistentStraggler(Devrajet
al.,2024/2025)
StragglARtargetstheprimaryweaknessofRAR:stragglersensitivity.Itimplementsa"parallel
All-Reduce"thatexploitsthenaturalvariationinGPUexecutiontimes.
●Contribution:Proposedanasymmetriccollectivewhere"fast"nodesperformapartial
Reduce-Scatterwhilewaitingforthestraggler,resultingina2timestheoreticalspeedup.

●TechnicalDetail:Onan8-GPUserver,StragglARyieldeda22%improvementover
state-of-the-artNCCLalgorithmsinthepresenceofapersistentstraggler.
8.Geo-DistributedMachineLearningwithGaia(Hsiehetal.,2017)
Gaiaaddressestheextremecommunicationconstraintsoftrainingmodelsacrossmultipledata
centersovertheWideAreaNetwork(WAN).
●Contribution:Introducedthe"SignificanceFilter,"whichonlycommunicatesgradient
updatesthatexceedathresholdofsignificance,reducingWANtrafficbyupto95%.
●TechnicalDetail:Usesamirror-clocksynchronizationmodeltoensurethatdifferent
datacentersdonotdrifttoofarapart,balancinglocalautonomywithglobalconsistency.
9.Pipe-SGD:APipelinedDistributedTrainingAlgorithm(Songetal.,2019)
Pipe-SGDintegratesRingAll-Reducewithpipeliningtohidecommunicationcostsin
decentralizedsystems.
●Contribution:Achievedhighperformancebyallowingthecommunicationofiterationk
tooverlapwiththecomputationofiterationk+1.
●TechnicalDetail:Provedthatthestalenessintroducedbypipeliningremainsconstant,
allowingforstableconvergenceifthedelayisaccountedforinthelearningrate.
10.EDiT:EfficientDistributedTrainingforLLMs(Zhuetal.,2025)
EDiTfocusesonthetrainingoflargelanguagemodelsusingacombinationofLocalSGDand
modelsharding.
●Contribution:Introducedlayer-wiseparametersynchronizationduringtheforwardpass,
effectivelyoverlappingthe"fetch"offuturelayerparameterswithcurrentlayer
computation.
●TechnicalDetail:Employsa"Pseudo-GradientPenalty"strategytodetectandsideline
anomalousworkercontributionsbasedontheirgradientnorms,improvingstabilityin
heterogeneousenvironments.
11.Strategy-Switch:FromAll-ReducetoParameterServer(2025)
Thispaperexploresahybridtrainingparadigmthattransitionsfromdecentralizedtocentralized
architecturesastrainingprogresses.
●Contribution:DemonstratedthatinitiatingtrainingwithRAR(highaccuracy)and
switchingtoasynchronousPS(highspeed)afterthemodelstabilizescanyielda1.14x
speedup.
●TechnicalDetail:Definedanempiricalrulebasedonvalidationlossstabilitytotrigger
theswitch,ensuringthattheaccuracylossiskeptbelow0.1%.

ComparativeSummaryTableofPeer-ReviewedLiterature
PaperArchitectureKey
## Mechanism
SynchronizationScaling
## Limit
## Primary
## Bottleneck
## Lietal.
## (2014)
## Centralized
## PS
Key-Value
## Store,
## Vector
## Clocks
Asynchronous/SSP10^{12}
## Parameters
## Server
## Bandwidth
(Many-to-One)
## Sergeev
etal.
## (2018)
## Decentralized
## Ring
## Ring
All-Reduce,
## Tensor
## Fusion
BulkSynchronousO(N)
## Bandwidth
Stragglers,High
Latency(O(N))
## Goyalet
al.(2017)
## Decentralized
## Ring
LinearLR
## Scaling,
## Warmup
Synchronous256GPUsGeneralization
(BatchSize
## Cap)
## Lianet
al.(2017)
DecentralizedPeer-to-Pee
rGossip
SynchronousLinear
## Speedup
## Topology
SpectralGap
## Zhanget
al.(2015)
## Centralized
## PS
## Elastic
## Force,
## Slack
## Models
AsynchronousComm-Effic
iency
## Hyperparameter
## Sensitivity

## Jianget
al.(2020)
## Hybrid
PS/Ring
## CPU
## Summation,
## Pipelining
PrioritySync256+GPUsNICBandwidth
## Mismatch
## Devrajet
al.(2025)
AdaptiveRingStraggler-A
ware
## Collective
AsymmetricSync2x
## Speedup
## Straggler
## Identity
## Persistence
## Hsiehet
al.(2017)
Geo-Distribut
edPS
## Significance
## Filtering
MirrorClockWAN
(Global)
HighNetwork
## Latency
## Songet
al.(2019)
## Pipelined
## Ring
Comm-Com
pOverlap
ConstantDelayPipelined
## Steps
## Gradient
## Staleness
## Zhuetal.
## (2025)
## Sharded
LocalSGD
## Layer-wise
Forward-Sy
nc
PeriodicLLMScaleMemoryfor
ShardReplicas
## Strategy-
## Switch
## (2025)
HybridDynamic
## Architecture
## Swap
Phase-Based1.14x
## Speedup
## Manual
## Transition
## Thresholds
