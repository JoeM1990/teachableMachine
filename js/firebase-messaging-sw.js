importScripts('https://www.gstatic.com/firebasejs/8.10.0/firebase-app.js');
importScripts('https://www.gstatic.com/firebasejs/8.10.0/firebase-messaging.js');


  // Initialize Firebase
  var config = {
    apiKey: "AIzaSyA5xZXI_dCPXnl3xLj30qAY1YYhKMeaHZA",
    authDomain: "pose-estimation-28d64.firebaseapp.com",
    projectId: "pose-estimation-28d64",
    storageBucket: "pose-estimation-28d64.appspot.com",
    messagingSenderId: "663567380839",
    appId: "1:663567380839:web:8a437c1189c36f2bba3fc2"
  };

  firebase.initializeApp(config);
  const messaging = firebase.messaging();