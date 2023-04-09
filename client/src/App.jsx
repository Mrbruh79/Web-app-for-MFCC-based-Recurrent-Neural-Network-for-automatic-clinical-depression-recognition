import "./App.css";
import { useState, useRef } from "react";
import AudioRecorder from "./components/AudioRecorder";
import Navigation from "./components/Navigation";
import HomePage from "./components/HomePage";
import Slid from "./components/Slid";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import AudioUpload from "./components/audioUpload";
const App = () => {
      return (
        <div className=" flex flex-col  h-screen">
        <Navigation />
      
       {/* <HomePage />  */}
       <h2 className="text-center text-white font-Roboto font-bold">Please upload your answers as an audio file </h2>
       <Slid />
       <AudioRecorder/>
       {/* <AudioUpload /> */}
       </div>
       
       
    );
};
export default App;