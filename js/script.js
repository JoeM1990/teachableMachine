/*
  Author : Joe MONKILA
*/

const STATUS = document.getElementById('status');
const VIDEO = document.getElementById('webcam');

let getlocal;

const REMOTE = document.getElementById('remote');



const RESULT = document.getElementById('result');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = [];
const OKK = document.getElementById('ok');

OKK.addEventListener('click', shareVideo);



//firebase.initializeApp(firebaseConfig);

//const db = firebase.firestore();

let pc1;
let pc2;
const offerOptions = {
  offerToReceiveAudio: 1,
  offerToReceiveVideo: 1
};

// const configuration = {'iceServers': [{'urls': 'stun:stun.l.google.com:19302'}]}
// const peerConnection = new RTCPeerConnection(configuration);


//var messagesRef = firebase.database().ref('status');

// let poseNet;
// let poses = [];

ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// Just add more buttons in HTML to allow classification of more classes of data!
let dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);
  // For mobile.
  dataCollectorButtons[i].addEventListener('touchend', gatherDataForClass);

  // Populate the human readable names for classes.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}


let mobilenet = undefined;
let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;


async function loadMobileNetFeatureModel() {

    const URL = 'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';
  
    // const URL2 = 'model/model.json';

    //mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});

    mobilenet = await tf.loadGraphModel(URL, {fromTFHub: true});

    STATUS.innerText = 'API chargé avec success';
  
    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
      let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
      console.log(answer.shape);
    });
  }
  
  // Call the function immediately to start loading.
  loadMobileNetFeatureModel();

  let model = tf.sequential();
  model.add(tf.layers.dense({inputShape: [1024], units: 128, activation: 'relu'}));
  model.add(tf.layers.dense({units: CLASS_NAMES.length, activation: 'softmax'}));
  
  model.summary();
  
  // Compile the model with the defined optimizer and specify a loss function to use.
  model.compile({
    // Adam changes the learning rate over time which is useful.
    optimizer: 'adam',
    // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
    // Else categoricalCrossentropy is used if more than 2 classes.
    loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy': 'categoricalCrossentropy',
    // As this is a classification problem you can record accuracy in the logs too!
    metrics: ['accuracy']
  });

  function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }
  
  function enableCam() {
    if (hasGetUserMedia()) {
      // getUsermedia parameters.
      const constraints = {
        video: true,
        width: 640,
        height: 480
      };
  
      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
       
        VIDEO.srcObject = stream;

        getlocal=stream;

        VIDEO.addEventListener('loadeddata', function() {
          videoPlaying = true;
          //ENABLE_CAM_BUTTON.classList.add('removed');
          ENABLE_CAM_BUTTON.remove();

          // poseNet = ml5.poseNet(VIDEO, modelLoad);
          // poseNet.on('pose', function(results) {
          //   poses = results;
          // });

          //test();
  
          
        });
      });
    } else {
      console.warn('getUserMedia() is not supported by your browser');
    }
  }

  // function modelLoad(){
  //   console.log('Model charge');
  // }

  function gatherDataForClass() {
    let classNumber = parseInt(this.getAttribute('data-1hot'));
    gatherDataState = (gatherDataState === STOP_DATA_GATHER) ? classNumber : STOP_DATA_GATHER;
    dataGatherLoop();
  }

  function dataGatherLoop() {
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
      let imageFeatures = tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT,
            MOBILE_NET_INPUT_WIDTH], true);
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
      });
  
      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);
  
      // Intialize array index element if currently undefined.
      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      examplesCount[gatherDataState]++;
  
      STATUS.innerText = '';

      for (let n = 0; n < CLASS_NAMES.length; n++) {
        STATUS.innerText += CLASS_NAMES[n] + ' Nombre de données: ' + examplesCount[n] + '    |    ';
      }
      window.requestAnimationFrame(dataGatherLoop);
    }
  }

  async function trainAndPredict() {
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);
  
    let results = await model.fit(inputsAsTensor, oneHotOutputs, {shuffle: true, batchSize: 5, epochs: 10,
        callbacks: {onEpochEnd: logProgress} });
  
    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict = true;
    predictLoop();
  }
  
  function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
  }

  function reset() {
    predict = false;
    examplesCount.length = 0;
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    trainingDataInputs.length = 0;
    trainingDataOutputs.length = 0;

    STATUS.innerText = 'Aucune donnée';
    RESULT.innerText = '';
  
    console.log('Tensors in memory: ' + tf.memory().numTensors);
  }

  function predictLoop() {
    if (predict) {
      tf.tidy(function() {
        let videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor,[MOBILE_NET_INPUT_HEIGHT,
            MOBILE_NET_INPUT_WIDTH], true);
  
        let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
        let prediction = model.predict(imageFeatures).squeeze();
        let highestIndex = prediction.argMax().arraySync();
        let predictionArray = prediction.arraySync();

        console.log(CLASS_NAMES[highestIndex]);
  
        STATUS.innerText = '';

        if(CLASS_NAMES[highestIndex] == 'Class 1'){
          RESULT.innerText =  " L'enfant est entrain d'étudier | " + 'Prédiction ' +' avec ' + Math.floor(predictionArray[highestIndex] * 100) + '% comme valeur';

          //var newMessageRef = messagesRef.push();
          //newMessageRef.delete();
          //newMessageRef.set({state: 0});

          // db.collection("status")
          // .get()
          // .then(res => {
          //   res.forEach(element => {
          //     element.ref.delete();
          //   });
          // });
          
          // db.collection('status').add({
          //   state: 0
          //  });


        }else{
          RESULT.innerText =  " L'enfant fait autres choses | " + 'Prédiction '+' avec ' + Math.floor(predictionArray[highestIndex] * 100) + '% comme valeur';

          //var newMessageRef = messagesRef.push();
          //newMessageRef.setValue(null);
          //newMessageRef.set({state: 1});

          // db.collection("status")
          // .get()
          // .then(res => {
          //   res.forEach(element => {
          //     element.ref.delete();
          //   });
          // });


          // db.collection('status').add({
          //   state: 1
          //  });


        }
        
      
      });
  
      window.requestAnimationFrame(predictLoop);
    }
  }


  // function draw() {
  //   // let width = 640;
  //   // let height = 480;

  //   image(VIDEO, 0, 0, width, height);
  //   drawKeypoints();
  //   drawSkeleton();
  // }
  
  // function drawKeypoints()  {
    
  //   for (let i = 0; i < poses.length; i++) {
      
  //     let pose = poses[i].pose;
  //     for (let j = 0; j < pose.keypoints.length; j++) {
        
  //       let keypoint = pose.keypoints[j];
       
  //       if (keypoint.score > 0.2) {
  //         fill(255, 0, 0);
  //         stroke(255, 255, 255);
  //         strokeWeight(1);
  //         ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
  
          
  //       }
  
       
  //     }
  //   }
  // }
  
  // function drawSkeleton() {
    
  //   for (let i = 0; i < poses.length; i++) {
  //     let skeleton = poses[i].skeleton;
     
  //     for (let j = 0; j < skeleton.length; j++) {
  //       let partA = skeleton[j][0];
  //       let partB = skeleton[j][1];
  //       stroke(255, 255, 255);
  //       strokeWeight(5);
  //       line(partA.position.x, partA.position.y, partB.position.x, partB.position.y);
  //     }
  //   }
  // }

  let idUser;
  let conn;

  var peer = new Peer(
    'pc1Monkila',{
      host: 'https://joem1990.github.io/teachableMachine/',
      debug: 2
  }
  );

  peer.on('open', function () {
   idUser = peer.id;
   console.log(peer.id)
  });

  let idAnother = 'pc2Monkila';

  function sendMessage(){

    conn = peer.connect(idAnother, {reliable: true});

    conn.on('open', function(){
      console.log('Connexion open');
      conn.send('Hi');
    });

    //conn.send('Hi');

  }

  function shareVideo(){

    conn = peer.connect(idAnother, {reliable: true});

    conn.on('open', function(){
      console.log('Connexion open');
      conn.send('Hi');
    });

    const constraints = {
      video: true,
      width: 640,
      height: 480
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
      var call = peer.call(idAnother, stream);
        call.on('stream', function(remoteStream) {
        // Show stream in some video/canvas element.
        REMOTE.srcObject = remoteStream;
     });
    });
   
  }
  


  


