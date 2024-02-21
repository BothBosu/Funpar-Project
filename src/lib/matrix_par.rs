use rand::{thread_rng, Rng};
use rayon::prelude::*;

#[derive(Clone)]
pub struct MatrixPar {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl MatrixPar {
    pub fn zeros(rows: usize, cols: usize) -> MatrixPar {
        MatrixPar {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows]
        }
    }

    pub fn randoms(rows: usize, cols:usize) -> MatrixPar {
        let mut rng = thread_rng();
        let mut res = MatrixPar::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> MatrixPar {
        MatrixPar {
            rows: data.len(),
            cols: data[0].len(),
            data
        }
    }

    pub fn multiply(&mut self, other: &MatrixPar) -> MatrixPar {
        if self.cols != other.rows {
            panic!("Multiply by MatrixPar of incorrect dimensions.")
        }
        let mut res = MatrixPar::zeros(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }

                res.data[i][j] = sum;
            }
        }
        res
    }

    pub fn add(&mut self, other: &MatrixPar) -> MatrixPar {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Add by MatrixPar of incorrect dimensions.")
        }
        let mut res = MatrixPar::zeros(self.rows, self.cols);

        res.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, cell)| {
                *cell = self.data[i][j] + other.data[i][j];
            });
        });
        res
    }

    pub fn subtract(&mut self, other: &MatrixPar) -> MatrixPar {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Subtract by MatrixPar of incorrect dimensions.")
        }
        let mut res = MatrixPar::zeros(self.rows, self.cols);

        res.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, cell)| {
                *cell = self.data[i][j] - other.data[i][j];
            });
        });
        res
    }

    //Hadamand product or Element-wise multiplication
    pub fn dot_multiply(&mut self, other: &MatrixPar) -> MatrixPar {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Dot Multiply by MatrixPar of incorrect dimensions.")
        }
        let mut res = MatrixPar::zeros(self.rows, self.cols);

        res.data.par_iter_mut().enumerate().for_each(|(i, row)| {
            row.par_iter_mut().enumerate().for_each(|(j, cell)| {
                *cell = self.data[i][j] * other.data[i][j];
            });
        });

        res
    }

    // Apply dynamic function to all element in MatrixPar
    pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> MatrixPar {
        MatrixPar::from(
        (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|value| function(value)).collect())
                .collect()
        )    
    }

    pub fn transpose(&mut self) -> MatrixPar {
        let mut res = MatrixPar::zeros(self.cols, self.rows);  
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }
}