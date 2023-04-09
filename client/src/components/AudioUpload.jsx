import { faL } from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import React from "react";
import { useState } from "react";



const AudioUpload = (props) =>{
    const uid=props.uid;

    const [file,setFile]= useState("");
    const [uploaded,setUploaded]=useState(false);

    const onFileChange =  (event) =>{
        // Create an object of formData
      setFile(event.target.files[0]);
    }

    const onFileUpload = async (e) =>{
        e.preventDefault();
        const formdata = new FormData();

        formdata.append("audioFile",file,"audio.weba");
        formdata.append('uid',uid); 
        const response = await axios.post('http://localhost:5000/audio', formdata , {
            headers:{
                "Content-Type":"audio/webm"
            }
        })
         
      
        if (response.ok) {
          console.log(response);
          setUploaded(true);
        } else {
          console.error('Error uploading audio file.');
          console.log(response.body);
          setUploaded(true);
        }
    }
    


    return (
        <div>
                <input type="file" onChange={(event)=>{setFile(event.target.files[0])}}/>
                <button type="submit"onClick={onFileUpload} className="text-white">
                  Upload!
                </button>
                {uploaded?<p>FIle uploaded!</p>:null}
    </div>

    )
    
}

export default AudioUpload;