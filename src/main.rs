pub mod network; // Get simulation.rs file
mod trainer;

extern crate glutin_window;
extern crate graphics;
extern crate opengl_graphics;
extern crate piston;

use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};

use piston::{ButtonEvent, RenderEvent};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, UpdateArgs, UpdateEvent, Button, ButtonState, Key, MouseCursorEvent};
use piston::window::WindowSettings;
use rand::Rng;

// Globals:
static BLACK: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

struct App {
    gl: GlGraphics, // OpenGL drawing backend.
    trainer : trainer::Trainer,
    score_curve : Vec<f64>
}


impl App {

    fn render(&mut self, args: &RenderArgs) {
        use graphics::*;

        // Create a square for drawing nodes
        let square = rectangle::square(0.0, 0.0, 10.0);

        // Create references to access them in the draw call
        let current_net = &self.trainer.net;
        let current_score_curve = self.score_curve.clone();

        self.gl.draw(args.viewport(), |c, gl| {
            clear(BLACK, gl); // Clear the screen.

            let transform = c.transform.trans(50.0, 50.0);

            // Get biggest layer for scaling
            let mut max_layer_size = 0;
            for l in 0..current_net.nodes.len() {
                if current_net.nodes[l].len() > max_layer_size {
                    max_layer_size = current_net.nodes[l].len();
                }
            }
            
            // Draw Graph
            for l in 0..current_net.nodes.len() {
                for n in 0..current_net.nodes[l].len() {
                    // Draw weights as lines
                    if l != current_net.nodes.len()-1 {
                        for nn in 0..current_net.nodes[l+1].len() {
                            line([current_net.weights[l][n][nn] as f32,
                                current_net.weights[l][n][nn] as f32,
                                current_net.weights[l][n][nn] as f32,
                                1.0], 0.4, [
                                50.0+(l as f64/(current_net.nodes.len() as f64)) * 400.0 + 5.0,
                                50.0+n as f64/(max_layer_size as f64) * 400.0 + 5.0,
                                50.0+((1+l) as f64/(current_net.nodes.len() as f64)) * 400.0 + 5.0,
                                50.0+nn as f64/(max_layer_size as f64) * 400.0 + 5.0
                            ], c.transform, gl);
                        }
                    }
                    
                    // Draw nodes
                    let node_color: [f32; 4] = [current_net.nodes[l][n] as f32,current_net.nodes[l][n] as f32,current_net.nodes[l][n] as f32,1.0];
                    rectangle(node_color, square, transform.
                        trans((l as f64/(current_net.nodes.len() as f64)) * 400.0, 
                             n as f64/(max_layer_size as f64) * 400.0),
                    gl);
                }
            }

            // Draw learning curve
            for i in 0..current_score_curve.len()-1 {
                // Draw lines between generation scores
                line([1.0; 4], 0.4, [
                    i as f64 * 5.0,
                    -current_score_curve[i] as f64 * 100.0,
                    (i+1) as f64 * 5.0,
                    -current_score_curve[i+1] as f64 * 100.0
                ], c.transform.trans(50.0, 500.0), gl);

                // Draw points at each generation point
                rectangle([1.0, 0.0, 0.0, 1.0], rectangle::square(0.0, 0.0, 2.0), c.transform.trans(50.0, 500.0).trans(i as f64 * 5.0, -current_score_curve[i] as f64 * 100.0).trans(-1.0, -1.0), gl);
            }
        });
    }

    /// Internal update loop
    fn update(&mut self, _args: &UpdateArgs) {
        
    }

    fn start(&mut self) {
        self.trainer.net = network::NeuralNetwork::new(vec![2,10, 10,2]);
        self.trainer.clear_training_data();
        self.trainer.add_training_data(vec![0.0, 0.0], vec![1.0, 1.0]);
        self.trainer.add_training_data(vec![1.0, 1.0], vec![0.0, 0.0]);
        self.trainer.add_training_data(vec![0.0, 1.0], vec![1.0, 0.0]);
        self.trainer.add_training_data(vec![1.0, 0.0], vec![0.0, 1.0]);
        self.score_curve = self.trainer.train_genetic_algorithm(20, 100, 1.0, 0.97);        
    }
}

/// Entrance Point to the program
fn main() {    
    initialize_graphics();
}

/// Initialize the window and graphcis library. Then start the main loop.
fn initialize_graphics() {
    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create an Glutin window.
    let mut window: Window = WindowSettings::new("spinning-square", [800, 800])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    
    
    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        trainer: trainer::Trainer::new(),
        score_curve : vec![]
    };

    app.start();
    
    let mut events = Events::new(EventSettings::new());
    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }

        if let Some(args) = e.update_args() {
            app.update(&args);
        }

        if let Some(_pos) = e.mouse_cursor_args() {
            //println!("{0} - {1}", pos[0] as f64, pos[1] as f64);
        }

        if let Some(k) = e.button_args() {
            if k.state == ButtonState::Press {
                match k.button {
                    Button::Keyboard(Key::D1) => {
                        let mut rng = rand::thread_rng();
                        app.trainer.net.set_inputs(vec![rng.gen_range(-1.0, 1.0), rng.gen_range(-1.0, 1.0)]);
                        app.trainer.net.calculate_network();
                    },
                    _ => (),
                }
            } else if k.state == ButtonState::Release {
                match k.button {
                    Button::Mouse(piston::MouseButton::Left) => {
                        // println!("{}", "LEFT");
                    },
                        _ => (),
                }
            }
        }
    }
}