use std::fs::File;
use std::io::Read;

pub fn load_images(path: &str) -> Vec<Vec<f64>> {
    let mut file = File::open(path).unwrap();
    let mut int_buffer = [0; 4];

    // Check magic bytes
    file.read_exact(&mut int_buffer).unwrap();
    let magic_num = u32::from_be_bytes(int_buffer);
    assert_eq!(2051, magic_num);

    // Get length of data
    file.read_exact(&mut int_buffer).unwrap();
    let size = u32::from_be_bytes(int_buffer);

    // List of images with bias parameter
    let mut images = Vec::<Vec<f64>>::with_capacity(size as usize);
    let mut buffer = [0; 4704]; // 6 images

    while let Ok(bytes_read) = file.read(&mut buffer) {
        if bytes_read == 0 {break;}

        for i in 0..bytes_read/784 {
            let mut vec = vec![1.0; 785];

            // Copy into vector
            for j in 0..784 {
                vec[j+1] = buffer[i*784+j] as f64 / 255.0;
            }

            images.push(vec);
        }
    }

    images
}

pub fn load_labels(path: &str) -> Vec<Vec<f64>> {
    let mut file = File::open(path).unwrap();
    let mut int_buffer = [0; 4];

    // Check magic bytes
    file.read_exact(&mut int_buffer).unwrap();
    let magic_num = u32::from_be_bytes(int_buffer);
    assert_eq!(2049, magic_num);

    // Get length of data
    file.read_exact(&mut int_buffer).unwrap();
    let size = u32::from_be_bytes(int_buffer);

    // List of properly formatted labels
    let mut labels = Vec::<Vec<f64>>::with_capacity(size as usize);
    let mut buffer = [0; 4096];

    while let Ok(bytes_read) = file.read(&mut buffer) {
        if bytes_read==0 {break;}

        for i in 0..bytes_read {
            let label = u8::from_be(buffer[i]);

            assert!(label <= 9);
            let mut vec = vec![0.0; 10];
            vec[label as usize] = 1.0;
            labels.push(vec);
        }
    }

    labels
}