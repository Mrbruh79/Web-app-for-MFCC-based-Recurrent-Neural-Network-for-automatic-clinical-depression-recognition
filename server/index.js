const express = require('express');
const cors = require('cors');
const fs = require('fs');
const wav = require('wav'); 
const app = express();
const {spawn}  = require('child_process');
app.use(cors());
const fileUpload = require('express-fileupload');


const spawnModel =  async (uid)=> {
  let datatoSend;

  const python = spawn('python', ['Model.py', `${uid}.wav`]);
  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...');
    const newData = JSON.parse(data);
    datatoSend=newData;
    console.log(dataToSend);
   });
   // in close event we are sure that stream from child process is closed
   python.on('exit', (code,signal) => {
   console.log(`child process sexit with code ${code}`);
   // send data to browser
    datatoSend=JSON.parse(datatoSend);
    return datatoSend;
   });
} 

app.use(fileUpload());

app.post('/audio', async (req, res) => {
    try {
        console.log(req.files);
        const audioFile=req.files.audioFile;
        const filename=req.body.uid;
        console.log(audioFile);
         audioFile.mv(__dirname+'/uploads2/'+filename,(err)=>{
             console.log(err);
         })
         res.send('Yes');
          
  } catch (err) {
    console.error(err);
    res.status(500).send('Server error.');
  }
  
});


app.listen(5000, () => {
  console.log('App listening at port 5000!');
});

app.post('/python',  async (req, res)=> {
  try {
    const uid=req.body.uid;
    let datatoSend;

    const python = spawn('python', ['Model.py', `${uid}.wav`]);
    python.stdout.on('data', function (data) {
      console.log('Pipe data from python script ...');
      console.log(data);
      datatoSend = JSON.parse(data);
     });
     // in close event we are sure that stream from child process is closed
     python.on('exit', (code,signal) => {
     console.log(`child process sexit with code ${code}`);
     // send data to browser
      res.send(datatoSend);
     });
     
  } catch(err) {
    res.send(err);
  }
})

