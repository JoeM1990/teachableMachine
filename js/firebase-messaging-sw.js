importScripts("https://www.gstatic.com/firebasejs/5.9.4/firebase-app.js");
importScripts("https://www.gstatic.com/firebasejs/5.9.4/firebase-messaging.js");

firebase.initializeApp({
	apiKey: "AIzaSyA5xZXI_dCPXnl3xLj30qAY1YYhKMeaHZA",
    authDomain: "pose-estimation-28d64.firebaseapp.com",
    projectId: "pose-estimation-28d64",
    storageBucket: "pose-estimation-28d64.appspot.com",
    messagingSenderId: "663567380839",
    appId: "1:663567380839:web:8a437c1189c36f2bba3fc2"
});

const messaging = firebase.messaging();

messaging.setBackgroundMessageHandler(function(payload) {
  const promiseChain = clients
    .matchAll({
      type: "window",
      includeUncontrolled: true
    })
    .then(windowClients => {
      for (let i = 0; i < windowClients.length; i++) {
        const windowClient = windowClients[i];
        windowClient.postMessage(payload);
      }
    })
    .then(() => {
      return registration.showNotification("my notification title");
    });
  return promiseChain;
});

self.addEventListener('notificationclick', function(event) {
  
});