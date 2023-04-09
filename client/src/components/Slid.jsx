import React from "react";

import Slider from "react-slick";



const Slid = () =>{
    const settings = {
      dots: true,
      infinite: true,
      speed: 500,
      slidesToShow: 1,
      slidesToScroll: 1,
        arrows:true,
        autoplay:true,
        pauseOnHover: true,
        appendDots: dots => (
          <div
            style={{
              backgroundColor: "#050a18",
              
            }}
          >
            <ul > {dots} </ul>
          </div>
        ),
        customPaging: i => (
          <div
            style={{
              width: "30px",
              color: "white",
              border: "1px #050a18 solid",
              marginTop:"2rem",
              paddingRight:"4rem "
            }}
          >
            {i + 1}
          </div>
        )
      };
      return(
      <div className=" text-center m-10 text-slate-50 w-1/2 mx-auto p-20">
      <Slider {...settings}>
      <div>
      <img src="questions\Slide1.JPG" className="m-auto max-h-96 bg-transparent aspect-video"/>
      </div>
      <div>
      <img src="questions\Slide2.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide3.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide4.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide5.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide6.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide7.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide8.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide9.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide10.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide11.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide12.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
      <div>
      <img src="questions\Slide13.JPG " className="m-auto max-h-96 aspect-video" />
      </div>
    </Slider>
    </div>)

}

export default Slid;