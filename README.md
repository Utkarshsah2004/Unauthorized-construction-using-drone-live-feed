# Monitoring Unauthorized Construction

## Overview
This repository contains the implementation and resources for "Online Monitoring of Unauthorized Construction Across the City," developed as part of **Smart India Hackathon 2024**. The project focuses on detecting and verifying unauthorized constructions using advanced image processing and machine learning techniques. Additionally, this solution leverages drone live feed data for real-time monitoring and analysis of unauthorized construction activities.

**Title:** Online Monitoring of Unauthorized Construction Across the City  
**Theme:** Robotics and Drones  

---

## Features
- Real-time analysis using live drone feed.
- Detection of rooftops and changes in area using image segmentation.
- Verification of changes using 3D models and historical data.
- Unauthorized construction alerts with notifications.
- Dynamic clustering for precise area detection.
- Integration with government and private land data.

---

## Technical Approach
### Programming Languages and Libraries
- **Programming Language:** Python
- **Frontend:** React (Three.js for 3D visualization)
- **Backend:** Django, Flask, Express
- **Libraries:**
  - TensorFlow
  - OpenCV
  - Matplotlib
  - Keras
  - NumPy

### Key Components
1. **Real-Time Feed Integration:**
   - Processes live drone feed for real-time construction detection.

2. **Image Segmentation:**
   - Utilizes DeepLabV3+ with ResNet50V2 backbone for semantic segmentation.
   - Converts top-view images to binary format for change detection.

3. **Depth Estimation:**
   - Calculates depth for each pixel and compares with historical data.

4. **Hu Moments:**
   - Analyzes shapes and detects discrepancies using log-scaled Hu Moments.

5. **XNOR Operations:**
   - Compares binary images to identify changes in structure.

6. **Clustering:**
   - Dynamic clustering to refine detection and ensure accuracy.

---

## File Descriptions
### Code Files
1. **1_rooftop.py:**
   - Handles rooftop detection using a pre-trained DeepLabV3+ model.
   - Provides functionality to overlay segmentation results and display images.
   - ![image alt](https://github.com/Utkarshsah2004/Unauthorized-construction-using-drone-live-feed/blob/main/Drone%20image.png?raw=true)
   - ![image alt](https://github.com/Utkarshsah2004/Unauthorized-construction-using-drone-live-feed/blob/main/Rooftop%20detection.png?raw=true)
2. **2_1_Area_cal.py:**
   - Implements region-growing algorithms to calculate area of clusters.
   - Assigns meaningful names to clusters and displays results visually.
   
3. **2_2_HuMoments2.py:**
   - Extracts and logs Hu Moments to analyze image shapes.
   - Provides insights into object contours and shape variations.

4. **3_XNOR.py:**
   - Performs bitwise XNOR operation on binary images to detect differences.
   - Highlights discrepancies in red for easy visualization.

### Presentation
- **SIH 2024 FINAL.pptx:** A detailed presentation covering the problem statement, solution approach, technical feasibility, and potential impact.

---

## Usage
1. **Setup Environment:**
   - Install required libraries: `pip install tensorflow opencv-python matplotlib keras numpy`
   - Clone this repository: `git clone https://github.com/yourusername/unauthorized-construction-monitoring`
   - Navigate to the directory: `cd unauthorized-construction-monitoring`

2. **Run the Scripts:**
   - Update file paths in scripts for your data.
   - Execute individual Python scripts for specific functionalities:
     ```bash
     python 1_rooftop.py
     python 2_1_Area_cal.py
     python 2_2_HuMoments2.py
     python 3_XNOR.py
     ```

3. **View Results:**
   - Processed images, visualizations, and logs will be saved in the output directory.

---

## Proposed Solution Workflow
1. **Path Planning:**
   - Use mission planners for drone routes.
   - Collect images with consistent pixel alignment.

2. **Real-Time Monitoring:**
   - Analyze live drone feed for immediate detection of unauthorized activities.

3. **Depth Mapping:**
   - Estimate and save pixel depths; compare with historical data.

4. **Top View Detection:**
   - Segment and analyze top-view images for changes.

5. **Binary Operations and Clustering:**
   - Convert to binary format, perform XNOR operations, and refine clusters.

6. **Action on Discrepancies:**
   - Government land: Register for review.
   - Private land: Validate against records, generate alerts, and update data.

---

## Impact and Benefits
### Impact on Stakeholders
- **Regulatory Bodies:** Streamlined enforcement and better compliance.
- **Property Owners:** Timely notifications and detailed construction insights.
- **City Planners:** Accurate data and models for urban development.

### Benefits
- **Social:** Enhances trust with fair construction practices.
- **Economic:** Reduces inspection costs and creates new revenue streams.
- **Environmental:** Minimizes illegal constructions impacting ecosystems.

---

## Challenges and Solutions
1. **Weather Dependence:** Use weather-resistant drones and plan operations during optimal conditions.
2. **Data Privacy Concerns:** Implement robust encryption and secure storage.
3. **User Training:** Provide interactive tutorials and ongoing support.

---

## References
- [OpenCV FFT for Blur Detection](https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams)
- [Shape Matching using Hu Moments](https://learnopencv.com/shape-matching-using-hu-moments-c-python/)
- [Fourier Transform](https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm)

---

## Contributing
We welcome contributions to improve this project. Feel free to submit issues or pull requests.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

