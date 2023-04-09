const express = require('express');
const cors = require('cors');
const fs = require('fs');
const wav = require('wav'); 
const app = express();
const {spawn}  = require('child_process');
const ffmpeg = require('fluent-ffmpeg');
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

const ffmpegWebAtoWav = (filename)=> {
  const ffmp = spawn('ffmpeg', ['-i', `${__dirname}/uploads/${filename}.weba`, `${__dirname}/uploads/${filename}.wav`]);
  ffmp.stdout.on('data', (data)=> {
    console.log("stdout: " + data);
  });
  ffmp.stderr.on('data', (data)=>{
    console.log("stderr: " + data);
  })
  ffmp.on('close', ()=> {
    console.log('fonwf');
      res.status(200).send({filename: filename+'.weba', directory: 'uploads'}); 
  });
}

app.post('/audio', async (req, res) => {
    try {
        console.log(req.files);
        const audioFile=req.files.audioFile;
        const filename=req.body.uid;
        console.log(audioFile);
         audioFile.mv(__dirname+'/uploads/'+filename+'.weba',(err)=>{
          const ffmp = spawn('ffmpeg', ['-i', `${__dirname}/uploads/${filename}.weba`, `${__dirname}/uploads/${filename}.wav`]);
          ffmp.stdout.on('data', (data)=> {
            console.log("stdout: " + data);
          });
          ffmp.stderr.on('data', (data)=>{
            console.log("stderr: " + data);
          })
          ffmp.on('close', ()=> {
            console.log('fonwf');
              res.status(200).send({filename: filename+'.wav', directory: 'uploads'}); 
          });
         });
         console.log(filename);
         
          
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
    console.log('Received post request');
    console.log(uid);
    
    

    const python = spawn("C:\\Users\\Restandsleep\\anaconda3\\python.exe", ['Model.py', `uploads/${uid}.weba`]);
    python.stdout.on('data', function (data) {
      console.log('Pipe data from python script ...');
      console.log(data);
      datatoSend = JSON.parse(data);
     });
     // in close event we are sure that stream from child process is closed
     python.on('exit', (code,signal) => {
     console.log(`child process exit with code ${code}`);
     // send data to browser
      res.send(datatoSend);
     });
     
  } catch(err) {
    res.send(err);
  }
})

