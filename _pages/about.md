---
permalink: /
title: "Biography"
excerpt: "About me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
  -
---

I am a research fellow at the University of St Andrews, funded by an early career fellowship from the [London Mathematical Society](https://www.lms.ac.uk/) and the [Heilbronn Institute for Mathematical Research](https://heilbronn.ac.uk/).  

Before that, I completed a PhD in mathematics with the [analysis research group](http://www.mcs.st-andrews.ac.uk/pg/pure/Analysis/) at the University of St Andrews. This was funded through a scholarship from the [Carnegie Trust](https://www.carnegie-trust.org/) and supervised by [Dr Jonathan Fraser](http://www.mcs.st-andrews.ac.uk/~jmf32/) and
[Professor Kenneth Falconer](http://www.mcs.st-and.ac.uk/~kenneth/).   

I enjoy teaching and was recently made an [Associate Fellow of the Higher Education Academy (AFHEA)](https://stuartburrell.github.io/files/cert-afhea.pdf). In 2020, three students kindly nominated me for the 'Best Postgraduate Tutor' university teaching award.  

<head>
<style>
.timeline * {
  box-sizing: border-box;
}

/* Set a background color */
.timeline body {
  background-color: #474e5d;
  font-family: Helvetica, sans-serif;
}

/* The actual timeline (the vertical ruler) */
.timeline {
  position: relative;
  max-width: 1200px;
  margin: 0 auto;
}

/* The actual timeline (the vertical ruler) */
.timeline::after {
  content: '';
  position: absolute;
  width: 6px;
  background-color: black;
  top: 0;
  bottom: 0;
  left: 50%;
  margin-left: -3px;
}

/* Container around content */
.timeline .container {
  padding: 10px 40px;
  position: relative;
  background-color: inherit;
  width: 50%;
}

/* The circles on the timeline */
.timeline .container::after {
  content: '';
  position: absolute;
  width: 25px;
  height: 25px;
  right: -10px;
  background-color: white;
  border: 4px solid #FF9F55;
  top: 15px;
  border-radius: 50%;
  z-index: 1;
}

/* Place the container to the left */
.timeline .left {
  left: 0;
}

/* Place the container to the right */
.timeline .right {
  left: 50%;
}

/* Add arrows to the left container (pointing right) */
.timeline .left::before {
  content: " ";
  height: 0;
  position: absolute;
  top: 22px;
  width: 0;
  z-index: 1;
  right: 30px;
  border: medium solid white;
  border-width: 10px 0 10px 10px;
  border-color: transparent transparent transparent white;
}

/* Add arrows to the right container (pointing left) */
.timeline .right::before {
  content: " ";
  height: 0;
  position: absolute;
  top: 22px;
  width: 0;
  z-index: 1;
  left: 30px;
  border: medium solid white;
  border-width: 10px 10px 10px 0;
  border-color: transparent white transparent transparent;
}

/* Fix the circle for containers on the right side */
.timeline .right::after {
  left: 0px;
}

/* The actual content */
.timeline .content {
  padding: 20px 30px;
  background-color: white;
  position: relative;
  border-radius: 6px;
}

/* Media queries - Responsive timeline on screens less than 600px wide */
@media screen and (max-width: 600px) {
/* Place the timelime to the left */
  .timeline::after {
    left: 31px;
  }

/* Full-width containers */
  .timeline .container {
    width: 100%;
    padding-left: 70px;
    padding-right: 25px;
  }

/* Make sure that all arrows are pointing leftwards */
   .timeline .container::before {
    left: 60px;
    border: medium solid white;
    border-width: 10px 10px 10px 0;
    border-color: transparent white transparent transparent;
  }

/* Make sure all circles are at the same spot */
  .timeline .left::after, .right::after {
    left: 15px;
  }

/* Make all right containers behave like the left ones */
  .timeline .right {
    left: 0%;
  }
}
</style>
</head>

<div class="timeline">
  <div class="container left">
    <div class="content">
      <h2>2017</h2>
      <p>Lorem ipsum..</p>
    </div>
  </div>
  <div class="container right">
    <div class="content">
      <h2>2016</h2>
      <p>Lorem ipsum..</p>
    </div>
  </div>
</div>




## Contact details

School of Mathematics and Statistics  
The University of St Andrews  
St Andrews  
KY16 9SS  
Scotland  
Email: sb235@st-andrews.ac.uk
