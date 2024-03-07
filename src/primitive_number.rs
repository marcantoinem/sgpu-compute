pub trait PrimitiveNumber: Default + Copy + bytemuck::Pod {}

#[macro_export]
macro_rules! number {
    ( $( $x:ty ),* ) => {
        $(
            impl PrimitiveNumber for $x {}
        )*
    };
}

// f64 is ommited because of the lack of support in WGSL
number!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32);
