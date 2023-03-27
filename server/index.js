const express = require('express');
const cors = require('cors');
const getRawBody = require('raw-body');
const fs = require('fs');
const wav = require('wav'); 
const app = express();
app.use(cors());
const fileUpload = require('express-fileupload');


app.use(fileUpload());

app.post('/audio', async (req, res) => {
    try {
        console.log(req.files);
        const audioFile=req.files.audioFile;
        const filename=req.files.audioFile.name;
        console.log(audioFile);
         audioFile.mv(__dirname+'/uploads2/'+filename,(err)=>{
             console.log(err);
         })
   
  } catch (err) {
    console.error(err);
    res.status(500).send('Server error.');
  }
  
});

app.listen(5000, () => {
  console.log('App listening at port 5000!');
});