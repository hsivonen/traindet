// Copyright 2019 Mozilla Foundation. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use bzip2::bufread::BzDecoder;
use detector_char_classes::*;
use detone::IterDecomposeVietnamese;
use encoding_rs::DecoderResult;
use encoding_rs::Encoding;
use encoding_rs::BIG5;
use encoding_rs::IBM866;
use encoding_rs::WINDOWS_1253;

use encoding_rs::EUC_JP;
use encoding_rs::EUC_JP_INIT;
use encoding_rs::EUC_KR;
use encoding_rs::EUC_KR_INIT;
use encoding_rs::GBK;
use encoding_rs::GBK_INIT;
use encoding_rs::IBM866_INIT;
use encoding_rs::ISO_8859_13_INIT;
use encoding_rs::ISO_8859_2_INIT;
use encoding_rs::ISO_8859_4_INIT;
use encoding_rs::ISO_8859_5_INIT;
use encoding_rs::ISO_8859_6_INIT;
use encoding_rs::ISO_8859_7_INIT;

use encoding_rs::ISO_8859_8_INIT;
use encoding_rs::KOI8_U_INIT;
use encoding_rs::WINDOWS_1251;
use encoding_rs::WINDOWS_1252;
use encoding_rs::WINDOWS_1255;
use encoding_rs::WINDOWS_1257;

use encoding_rs::EncoderResult;
use encoding_rs::WINDOWS_1250;
use encoding_rs::WINDOWS_1250_INIT;
use encoding_rs::WINDOWS_1251_INIT;
use encoding_rs::WINDOWS_1252_INIT;
use encoding_rs::WINDOWS_1253_INIT;
use encoding_rs::WINDOWS_1254;
use encoding_rs::WINDOWS_1254_INIT;
use encoding_rs::WINDOWS_1255_INIT;
use encoding_rs::WINDOWS_1256;
use encoding_rs::WINDOWS_1256_INIT;
use encoding_rs::WINDOWS_1257_INIT;
use encoding_rs::WINDOWS_1258;
use encoding_rs::WINDOWS_1258_INIT;

use encoding_rs::WINDOWS_874_INIT;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use unic_normal::StrNormalForm;
use unicode_reader::CodePoints;

struct CharMap {
    // Highest is U+25A0 BLACK SQUARE
    arr: [u8; 9633],
}

impl CharMap {
    fn new(char_classes: &'static [&'static [char]], windows_encoding: &'static Encoding) -> Self {
        let mut ret = CharMap { arr: [0u8; 9633] };
        for (i, chars) in char_classes.iter().enumerate() {
            let class = i as u8;
            for &c in chars.iter() {
                ret.arr[c as usize] = class;

                let upper = if windows_encoding == WINDOWS_1254 && c == 'i' {
                    'İ'
                } else if c == 'ς' {
                    // Intentionally not handling final sigma to match
                    // detection-time mapping.
                    continue;
                } else {
                    let mut iter = c.to_uppercase();
                    let first = iter.next().unwrap();
                    if let Some(_) = iter.next() {
                        continue;
                    }
                    first
                };
                ret.arr[upper as usize] = class;
            }
        }
        // if !is_latin(windows_encoding) {
        //     for c in b'a'..=b'z' {
        //         ret.arr[c as usize] = 0xFE;
        //     }
        //     for c in b'A'..=b'Z' {
        //         ret.arr[c as usize] = 0xFE;
        //     }
        //     if windows_encoding == WINDOWS_1256 {
        //         for &c in ARABIC_FRENCH.iter() {
        //             ret.arr[c as usize] = 0xFE;
        //         }
        //     }
        // }
        ret
    }

    fn get(&self, c: char) -> u8 {
        let s = c as usize;
        if s < self.arr.len() {
            return self.arr[s];
        }
        0u8
    }
}

#[derive(Debug)]
struct EncodingClass {
    char_classes: &'static [&'static [char]],
    encodings: &'static [&'static Encoding],
    languages: &'static [&'static str],
    name: &'static str,
    space_divisor: f64,
    multiplier: f64,
}

impl EncodingClass {
    fn count(&'static self, dir: &Path) {
        let windows_encoding = self.encodings[0];

        let map = CharMap::new(self.char_classes, windows_encoding);

        let (ascii_classes, non_ascii_classes) = count_ascii_classes(self.char_classes);

        self.languages.par_iter().for_each(|lang| {
            eprintln!("Counting {:?}", lang);
            let corpus = find_file(dir, lang);
            let scores = count_one(
                &corpus,
                &map,
                ascii_classes,
                non_ascii_classes,
                windows_encoding,
                false,
            );
            let path = gen_count_path(dir, lang, false);
            write_scores(&path, &scores);

            if windows_encoding == WINDOWS_1258 {
                let corpus = find_file(dir, lang);
                let scores = count_one(
                    &corpus,
                    &map,
                    ascii_classes,
                    non_ascii_classes,
                    windows_encoding,
                    true,
                );
                let path = gen_count_path(dir, lang, true);
                write_scores(&path, &scores);
            }
        });
    }

    fn train(&'static self, dir: &Path) -> (Vec<f64>, &'static Self) {
        let windows_encoding = self.encodings[0];

        // let map = CharMap::new(self.char_classes, windows_encoding);

        let (ascii_classes, non_ascii_classes) = count_ascii_classes(self.char_classes);

        let language_scores = self
            .languages
            .iter()
            .map(|lang| {
                eprintln!("Training {:?}", lang);
                let count_path = gen_count_path(dir, lang, false);
                let mut scores = load_one(&count_path);
                divide_by_class_size(
                    &mut scores,
                    self.char_classes,
                    ascii_classes,
                    non_ascii_classes,
                    windows_encoding,
                    self.space_divisor,
                );

                multiply(&mut scores, self.multiplier);

                if windows_encoding == WINDOWS_1256 {
                    // Some letter pairs in Arabic get such high
                    // scores that it the corresponding byte pairs
                    // occur in a short string, pretty much anything
                    // gets classified as Arabic. Since in practice,
                    // single-byte Arabic encodings on the Web are
                    // rare, this intervention makes the detector
                    // less eager to decide Arabic.
                    // clamp(
                    //     &mut scores,
                    //     self.char_classes,
                    //     ascii_classes,
                    //     non_ascii_classes,
                    //     windows_encoding,
                    //     'ا',
                    //     'ئ',
                    // );
                }

                let mut max = 0.0f64;
                for &score in scores.iter() {
                    if score > max {
                        max = score;
                    }
                }

                println!("MAX {}: {:?}", lang, max);

                if windows_encoding == WINDOWS_1258 {
                    let mut vietnamese_scores = Vec::new();
                    vietnamese_scores.push(scores);
                    let count_path = gen_count_path(dir, lang, true);
                    let mut scores = load_one(&count_path);
                    divide_by_class_size(
                        &mut scores,
                        self.char_classes,
                        ascii_classes,
                        non_ascii_classes,
                        windows_encoding,
                        self.space_divisor,
                    );

                    let mut max = 0.0f64;
                    for &score in scores.iter() {
                        if score > max {
                            max = score;
                        }
                    }
                    vietnamese_scores.push(scores);

                    merge(vietnamese_scores)
                } else {
                    scores
                }
            })
            .collect();

        (merge(language_scores), self)
    }
}

static ENCODING_CLASSES: [EncodingClass; 11] = [
    // Vietnamese consumes the corpus twice, so put it first
    // to maximize parallelism.
    // In the `encodings` field, the Windows encoding comes first.
    EncodingClass {
        char_classes: &VIETNAMESE,
        encodings: &[&WINDOWS_1258_INIT],
        languages: &["vi"],
        name: "vietnamese",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &CENTRAL,
        encodings: &[&WINDOWS_1250_INIT, &ISO_8859_2_INIT],
        languages: &["pl", "hu", "sh", "cs", "ro", "sk", "hr", "sl", "bs"],
        name: "central",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &CYRILLIC,
        // IE and Chromium don't detect x-mac-cyrillic.
        encodings: &[
            &WINDOWS_1251_INIT,
            &KOI8_U_INIT,
            &ISO_8859_5_INIT,
            &IBM866_INIT,
        ],
        // kk, tt, tg, and os don't fit
        // mn uses mapping to uk letters
        languages: &["ru", "uk", "sr", "bg", "ce", "be", "mk", "mn"],
        name: "cyrillic",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &WESTERN,
        encodings: &[&WINDOWS_1252_INIT],
        // Intentionally omitting ASCII or almost-ASCII languages like en, nl, id, so, sw, various Malay-alphabet languages.
        // Per https://web.archive.org/web/20170916200715/http://www.microsoft.com/resources/msdn/goglobal/default.mspx
        // Albanian got windows-1250 from Microsoft, but it the characters overlap windows-1252, and the Google stats
        // say windows-1252. By putting Albanian here, it doesn't introduce misdetections between windows-1252 and
        // windows-1250.
        // Estonian gets windows-1257 on Windows. However, glibc and Solaris 2.6 had Estonian as an ISO-8859-1 locale.
        // Putting Estonian here avoids diluting the Baltic model with traits shared with German, Swedish, Finnish, and
        // Portuguese. In practice, Estonian itself gains significantly in detection accuracy by being in this model.
        languages: &[
            "sv", "de", "fr", "it", "es", "pt", "ca", "no", "fi", "eu", "da", "gl", "nn", "oc",
            "br", "lb", "ht", "ga", "an", "wa", "gd", "li", "sq", "et",
        ],
        name: "western",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &ICELANDIC,
        encodings: &[&WINDOWS_1252_INIT],
        // Intentionally omitting ASCII or almost-ASCII languages like en, nl, id, so, sw, various Malay-alphabet languages
        languages: &["is", "fo"],
        name: "icelandic",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &GREEK,
        encodings: &[&WINDOWS_1253_INIT, &ISO_8859_7_INIT],
        languages: &["el"],
        name: "greek",
        space_divisor: 1.0,
        multiplier: 1.3,
    },
    EncodingClass {
        char_classes: &TURKISH,
        encodings: &[&WINDOWS_1254_INIT],
        languages: &["tr", "az", "ku"],
        name: "turkish",
        space_divisor: 1.0,
        multiplier: 1.8,
    },
    EncodingClass {
        char_classes: &HEBREW,
        encodings: &[&WINDOWS_1255_INIT, &ISO_8859_8_INIT],
        languages: &["he", "yi"],
        name: "hebrew",
        space_divisor: 1.0,
        multiplier: 2.0,
    },
    EncodingClass {
        char_classes: &ARABIC,
        encodings: &[&WINDOWS_1256_INIT, &ISO_8859_6_INIT],
        languages: &["ar", "fa", "ur"],
        name: "arabic",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &BALTIC,
        encodings: &[&WINDOWS_1257_INIT, &ISO_8859_13_INIT, &ISO_8859_4_INIT],
        languages: &["lt", "lv"],
        name: "baltic",
        space_divisor: 1.0,
        multiplier: 1.0,
    },
    EncodingClass {
        char_classes: &THAI,
        encodings: &[&WINDOWS_874_INIT],
        languages: &["th"],
        name: "thai",
        space_divisor: 1.0,
        multiplier: 1.7,
    },
];

static CJK_ENCODINGS: [(&'static str, &'static Encoding, &'static str); 3] = [
    ("zh-hans", &GBK_INIT, "simplified"),
    // Big5 is distinctive without extra data
    // ("zh-hant", &BIG5_INIT, "traditional"),
    ("ja", &EUC_JP_INIT, "kanji"),
    ("ko", &EUC_KR_INIT, "hangul"),
];

fn write_scores(path: &Path, scores: &[u64]) {
    let bytes = {
        // Actually safe, since all bit patterns are valid values
        // for u8 and u64.
        let (head, bytes, tail) = unsafe { scores.align_to::<u8>() };
        assert_eq!(head.len(), 0);
        assert_eq!(tail.len(), 0);
        bytes
    };

    let file = File::create(path).expect("Unable to create output file.");
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes).unwrap();
}

fn gen_count_path(dir: &Path, lang: &str, bis: bool) -> PathBuf {
    let suffix = if bis { "_bis.counts" } else { ".counts" };
    let mut file_name = String::from(lang);
    file_name.push_str(suffix);
    dir.join(file_name)
}

fn find_file(dir: &Path, lang: &str) -> PathBuf {
    for entry in dir
        .read_dir()
        .expect("Reading the corpus directory failed.")
    {
        if let Ok(entry) = entry {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            if s.starts_with(lang) && s.ends_with(".bz2") {
                return entry.path();
            }
        }
    }
    eprintln!("Error: No corpus for: {}", lang);
    std::process::exit(-4);
}

fn count_ascii_classes(char_classes: &'static [&'static [char]]) -> (usize, usize) {
    let ascii_classes = char_classes
        .iter()
        .map(|c| c[0])
        .take_while(|c| *c < '\u{80}')
        .count();
    (ascii_classes, char_classes.len() - ascii_classes)
}

fn open_bzip2(path: &Path) -> impl Iterator<Item = char> {
    let dec = BzDecoder::new(BufReader::new(File::open(path).unwrap()));
    CodePoints::from(BufReader::new(dec)).map(|r| r.unwrap())
    // .take(50000) // XXX remove
}

fn merge(language_scores: Vec<Vec<f64>>) -> Vec<f64> {
    let mut iter = language_scores.into_iter();
    let mut ret = iter.next().unwrap();
    for vec in iter {
        assert_eq!(ret.len(), vec.len());
        for (r, v) in ret.iter_mut().zip(vec.into_iter()) {
            *r = f64::max(*r, v);
        }
    }
    ret
}

fn multiply(scores: &mut Vec<f64>, factor: f64) {
    for score in scores.iter_mut() {
        *score *= factor;
    }
}

fn clamp(
    scores: &mut Vec<f64>,
    classes: &'static [&'static [char]],
    ascii_classes: usize,
    non_ascii_classes: usize,
    windows_encoding: &'static Encoding,
    first_reference: char,
    second_reference: char,
) {
    let map = CharMap::new(classes, windows_encoding);
    let index = compute_index(
        map.get(first_reference) as usize,
        map.get(second_reference) as usize,
        ascii_classes,
        non_ascii_classes,
    )
    .unwrap();
    let cutoff = scores[index];
    for score in scores.iter_mut() {
        if *score > cutoff {
            *score = cutoff;
        }
    }
}

fn divide_by_class_size(
    scores: &mut Vec<f64>,
    classes: &'static [&'static [char]],
    ascii_classes: usize,
    non_ascii_classes: usize,
    windows_encoding: &'static Encoding,
    space_divisor: f64,
) {
    for (i, size) in classes.iter().map(|c| c.len()).enumerate() {
        let divisor = if i == 1 && windows_encoding == WINDOWS_1255 {
            // Don't divide Hebrew ASCII punctuation class
            continue;
        } else if i == 0 {
            space_divisor
        } else {
            if size == 1 {
                continue;
            }
            size as f64
        };
        for j in 0..classes.len() {
            if let Some(index) = compute_index(i, j, ascii_classes, non_ascii_classes) {
                scores[index] /= divisor;
            }
            if let Some(index) = compute_index(j, i, ascii_classes, non_ascii_classes) {
                scores[index] /= divisor;
            }
        }
    }
}

#[inline(always)]
fn compute_index(
    x: usize,
    y: usize,
    ascii_classes: usize,
    non_ascii_classes: usize,
) -> Option<usize> {
    if x == 0 && y == 0 {
        return None;
    }
    if x < ascii_classes && y < ascii_classes {
        return None;
    }
    if y >= ascii_classes {
        return Some(
            (ascii_classes * non_ascii_classes)
                + (ascii_classes + non_ascii_classes) * (y - ascii_classes)
                + x,
        );
    }
    Some(y * non_ascii_classes + x - ascii_classes)
}

fn compute_scores<I: Iterator<Item = char>>(
    iter: I,
    classes: &CharMap,
    ascii_classes: usize,
    non_ascii_classes: usize,
) -> Vec<u64> {
    let score_len = non_ascii_classes * non_ascii_classes + 2 * (ascii_classes * non_ascii_classes);
    let mut scores = Vec::new();
    scores.resize(score_len, 0u64);

    let mut prev = 0usize;

    for c in iter {
        let current = classes.get(c) as usize;
        if let Some(index) = compute_index(prev, current, ascii_classes, non_ascii_classes) {
            scores[index] += 1;
        }
        prev = current;
    }

    scores
}

fn convert_to_floats(scores: &[u64]) -> Vec<f64> {
    let total: u64 = scores.iter().sum();

    let mut float_scores = Vec::with_capacity(scores.len());
    let float_total = total as f64;
    for &score in scores {
        if score == 0 {
            // No instances of this character pair.
            // Mark as implausible.
            float_scores.push(std::f64::NAN);
        } else {
            let float_score = score as f64;
            float_scores.push(float_score / float_total);
        }
    }

    float_scores
}

fn load_u64(path: &Path) -> Vec<u64> {
    let bytes = std::fs::read(path).expect("Could not read counts");
    assert_eq!(bytes.len() % std::mem::size_of::<u64>(), 0);
    let mut ret = Vec::new();
    ret.resize(bytes.len() / std::mem::size_of::<u64>(), 0);
    // Actually safe, since all bit patterns are valid values
    // for u8 and u64.
    let (head, target, tail) = unsafe { ret.align_to_mut::<u8>() };
    assert_eq!(head.len(), 0);
    assert_eq!(tail.len(), 0);

    target.copy_from_slice(&bytes);

    ret
}

fn load_one(path: &Path) -> Vec<f64> {
    let scores = load_u64(path);
    convert_to_floats(&scores)
}

fn count_one(
    path: &Path,
    classes: &CharMap,
    ascii_classes: usize,
    non_ascii_classes: usize,
    encoding: &'static Encoding,
    orthographic_vietnamese: bool,
) -> Vec<u64> {
    let iter = open_bzip2(path);

    // if encoding == WINDOWS_1256 {
    //     // Map non-ASCII Latin to ASCII Latin
    //     compute_scores(
    //         iter.nfc().map(|c| {
    //             // XXX upper case
    //             if ARABIC_FRENCH.iter().find(|&&x| x == c).is_some() {
    //                 'a'
    //             } else {
    //                 c
    //             }
    //             c
    //         }),
    //         classes,
    //         ascii_classes,
    //         non_ascii_classes,
    //     )
    // } else
    if encoding == WINDOWS_1258 {
        // Decompose tones
        compute_scores(
            iter.nfc()
                .decompose_vietnamese_tones(orthographic_vietnamese),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1254 {
        // Map Azeri ə to ä.
        compute_scores(
            iter.nfc().map(|c| match c {
                'Ə' => 'Ä',
                'ə' => 'ä',
                _ => c,
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1250 {
        // Map Romanian comma-below characters to cedilla versions,
        // because legacy encodings (other than ISO-8859-16, which we
        // won't detect) only have the latter.
        compute_scores(
            iter.nfc().map(|c| match c {
                'ț' => 'ţ',
                'ș' => 'ş',
                'Ț' => 'Ţ',
                'Ș' => 'Ş',
                _ => c,
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else if encoding == WINDOWS_1251 {
        // Map Mongolian characters to Ukranian substitutes
        compute_scores(
            iter.nfc().map(|c| match c {
                'Ү' => 'Ї',
                'ү' => 'ї',
                'Ө' => 'Є',
                'ө' => 'є',
                _ => c,
            }),
            classes,
            ascii_classes,
            non_ascii_classes,
        )
    } else {
        compute_scores(iter.nfc(), classes, ascii_classes, non_ascii_classes)
    }
}

fn further_than_epsilon(a: u8, b: u8, e: u8) -> bool {
    let delta = if a < b { b - a } else { a - b };
    delta > e
}

/// Finds the first class _after_ letters for Latin and the first
/// non-Latin letter class for non-Latin.
fn find_letter_boundary(encoding_class: &'static EncodingClass) -> u8 {
    if is_latin(encoding_class.encodings[0]) {
        for (i, chars) in encoding_class.char_classes.iter().enumerate() {
            let first = chars[0];
            if first == ' ' || first == '0' || first == 'a' {
                continue;
            }
            if !first.is_alphabetic() {
                return i as u8;
            }
        }
    } else {
        for (i, chars) in encoding_class.char_classes.iter().enumerate() {
            let first = chars[0];
            if first == ' ' || first == '0' || first == 'a' {
                continue;
            }
            if first.is_alphabetic() {
                return i as u8;
            }
        }
    }
    encoding_class.char_classes.len() as u8
}

fn suggest_merges(scores: &Vec<u8>, encoding_class: &'static EncodingClass) {
    let (ascii_classes, non_ascii_classes) = count_ascii_classes(encoding_class.char_classes);
    for e in 0..1 {
        for i in 0..encoding_class.char_classes.len() {
            'mid: for j in 0..i {
                if i >= ascii_classes && j < ascii_classes
                    || j >= ascii_classes && i < ascii_classes
                {
                    // Don't suggest merging ASCII and non-ASCII
                    continue;
                }
                for k in 0..encoding_class.char_classes.len() {
                    let i_index_1 = compute_index(i, k, ascii_classes, non_ascii_classes);
                    let j_index_1 = compute_index(j, k, ascii_classes, non_ascii_classes);
                    if let (Some(i_index), Some(j_index)) = (i_index_1, j_index_1) {
                        if further_than_epsilon(scores[i_index], scores[j_index], e) {
                            continue 'mid;
                        }
                    }
                    let i_index_2 = compute_index(k, i, ascii_classes, non_ascii_classes);
                    let j_index_2 = compute_index(k, j, ascii_classes, non_ascii_classes);
                    if let (Some(i_index), Some(j_index)) = (i_index_2, j_index_2) {
                        if further_than_epsilon(scores[i_index], scores[j_index], e) {
                            continue 'mid;
                        }
                    }
                }
                // Suggest
                println!(
                    "Epsilon: {:?}, {}, merge: {:?} & {:?}",
                    e,
                    encoding_class.name,
                    encoding_class.char_classes[i],
                    encoding_class.char_classes[j]
                );
            }
        }
    }
}

fn find_class(c: char, char_classes: &'static [&'static [char]]) -> Option<usize> {
    for (i, chars) in char_classes.iter().enumerate() {
        if chars[0] == c {
            return Some(i);
        }
    }
    None
}

fn force_implausibility_next_to_alphabetic(
    c: char,
    implausible_after_alphabetic: bool,
    byte_vec: &mut Vec<u8>,
    char_classes: &'static [&'static [char]],
    ascii_classes: usize,
    non_ascii_classes: usize,
    allow_next_to_ascii: bool,
) {
    if let Some(special) = find_class(c, char_classes) {
        for (i, chars) in char_classes.iter().enumerate() {
            if chars[0].is_alphabetic() && (!allow_next_to_ascii || chars[0] >= '\u{80}') {
                let (first, second) = if implausible_after_alphabetic {
                    (i, special)
                } else {
                    (special, i)
                };
                if let Some(index) = compute_index(first, second, ascii_classes, non_ascii_classes)
                {
                    byte_vec[index] = 255;
                }
            }
        }
    }
}

fn force_implausibility_next_to_self(
    c: char,
    byte_vec: &mut Vec<u8>,
    char_classes: &'static [&'static [char]],
    ascii_classes: usize,
    non_ascii_classes: usize,
) {
    if let Some(special) = find_class(c, char_classes) {
        if let Some(index) = compute_index(special, special, ascii_classes, non_ascii_classes) {
            byte_vec[index] = 255;
        }
    }
}

fn force_implausibility(byte_vec: &mut Vec<u8>, encoding_class: &'static EncodingClass) {
    // The use case for this stuff is primarily to avoid misdetecting windows-1250 as other Latin and to avoid misdetecting
    // Greek as Hebrew or Cyrillic.
    let (ascii_classes, non_ascii_classes) = count_ascii_classes(encoding_class.char_classes);
    // TODO: Force implausibility next to itself
    // '·', plus the classes below
    // Ordinal indicators are supposed to be preceded by a digit and be followed by space.
    force_implausibility_next_to_alphabetic(
        'ª',
        true,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_alphabetic(
        'ª',
        false,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_self(
        'ª',
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
    );
    // Boost after number for windows-1252

    // Implausible on either side of alphabetic
    force_implausibility_next_to_alphabetic(
        '¨',
        true,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_alphabetic(
        '¨',
        false,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_self(
        '¨',
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
    );
    // Implausible before alphabetic
    force_implausibility_next_to_alphabetic(
        '®',
        false,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_self(
        '®',
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
    );
    // Implausible after alphabetic
    force_implausibility_next_to_alphabetic(
        '©',
        true,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_self(
        '©',
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
    );
    // Implausible on either side of non-ASCII alphabetic
    force_implausibility_next_to_alphabetic(
        '¬',
        true,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        true,
    );
    force_implausibility_next_to_alphabetic(
        '¬',
        false,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        true,
    );
    force_implausibility_next_to_self(
        '¬',
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
    );
    // LRM and RLM are meant to be used after a punctuation character before space. Let's mark them implausible before and after
    // a letter. It's not technically wrong for them to appear next to a letter, but there's no need, and allowing it causes
    // misdetections of Greek as Hebrew.
    force_implausibility_next_to_alphabetic(
        '\u{200E}',
        true,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_alphabetic(
        '\u{200E}',
        false,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    force_implausibility_next_to_self(
        '\u{200E}',
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
    );
    // Final sigma
    force_implausibility_next_to_alphabetic(
        'ς',
        false,
        byte_vec,
        encoding_class.char_classes,
        ascii_classes,
        non_ascii_classes,
        false,
    );
    // To avoid misdetection of Turkish and Lithuanian, we
    // make the Icelandic and Faroese implausible in the
    // main windows-1252 model. (There's a separate model
    // for Icelandic and Fareose.)
    if encoding_class.name == "western" {
        force_implausibility_next_to_alphabetic(
            'þ',
            false,
            byte_vec,
            encoding_class.char_classes,
            ascii_classes,
            non_ascii_classes,
            false,
        );
        force_implausibility_next_to_alphabetic(
            'þ',
            true,
            byte_vec,
            encoding_class.char_classes,
            ascii_classes,
            non_ascii_classes,
            false,
        );
    }
    // Mark Vietnamese tones implausible on characters that aren't valid bases or space.
    if encoding_class.encodings[0] == WINDOWS_1258 {
        let tones = ['\u{0300}', '\u{0309}', '\u{0303}', '\u{0301}', '\u{0323}'];
        for &tone in tones.iter() {
            let tone_class_index = find_class(tone, encoding_class.char_classes).unwrap();
            for base_class_index in 0..encoding_class.char_classes.len() {
                match encoding_class.char_classes[base_class_index][0] {
                    'a' | 'ă' | 'â' | 'e' | 'ê' | 'i' | 'o' | 'ô' | 'ơ' | 'u' | 'ư' | 'y' => {
                    }
                    ' ' => {
                        if let Some(index) = compute_index(
                            base_class_index,
                            tone_class_index,
                            ascii_classes,
                            non_ascii_classes,
                        ) {
                            byte_vec[index] = 0;
                        }
                    }
                    _ => {
                        if let Some(index) = compute_index(
                            base_class_index,
                            tone_class_index,
                            ascii_classes,
                            non_ascii_classes,
                        ) {
                            byte_vec[index] = 255;
                        }
                    }
                }
            }
        }
    }
}

fn train_with_dir(dir: &Path, rs: &Path) {
    let float_scores = ENCODING_CLASSES
        .iter()
        .map(|c| c.train(dir))
        .collect::<Vec<(Vec<f64>, &'static EncodingClass)>>();
    // let mut max = 0.0f64;
    // for (vec, _) in float_scores.iter() {
    //     for &score in vec.iter() {
    //         if score > max {
    //             max = score;
    //         }
    //     }
    // }
    // Albanian and Breton are so unlike the others that
    // max doesn't make sense above. Italian, Finnish,
    // Nynorsk, Aragonese, Romanian, Luxembourgish, and Walloon
    // are somewhat of outliers, too. Let's hard-code 0.1,
    // which is around Portuguese, Spanish, and Bokmål.
    let max = 0.1;

    let mut scores = Vec::with_capacity(float_scores.len());
    for (vec, encoding_class) in float_scores.iter() {
        let mut byte_vec = Vec::new();
        byte_vec.resize(vec.len(), 0u8);
        for (b, f) in byte_vec.iter_mut().zip(vec.into_iter()) {
            if f.is_nan() {
                *b = 255;
            } else {
                let s = f64::floor((f / max) * 254.5) as u64;
                *b = if s > 254 { 254 } else { s as u8 };
            }
        }
        scores.push((byte_vec, *encoding_class));
    }

    for (byte_vec, encoding_class) in scores.iter_mut() {
        force_implausibility(byte_vec, encoding_class);
    }

    for (float_vec, encoding_class) in float_scores.iter() {
        let mut max = 0.0f64;
        for &score in float_vec.iter() {
            if score > max {
                max = score;
            }
        }

        println!("MAX {}: {:?}", encoding_class.name, max);
    }

    for (byte_vec, encoding_class) in scores.iter() {
        println!(
            "MAX byte {}: {:?}",
            encoding_class.name,
            byte_vec
                .iter()
                .map(|x| if *x == 255 { 0 } else { *x })
                .max()
                .unwrap()
        );
    }

    for (byte_vec, encoding_class) in scores.iter() {
        suggest_merges(&byte_vec, encoding_class);
    }

    // CJK

    let mut cjk_scores = Vec::new();

    for (lang, encoding, name) in CJK_ENCODINGS.iter() {
        cjk_scores.push((cjk_frequent(dir, lang, encoding), *name));
    }

    write_rs_file(rs, &scores, &cjk_scores);
}

fn cjk_frequent(dir: &Path, lang: &str, encoding: &'static Encoding) -> Vec<u16> {
    let (start, end) = cjk_bounds(lang);
    let count_path = gen_count_path(dir, lang, false);
    let counts = load_u64(&count_path);
    assert_eq!(counts.len(), end - start);

    let mut sortable: Vec<(usize, u64)> = counts
        .iter()
        .enumerate()
        .map(|(i, count)| (i + start, *count))
        .collect();
    sortable.sort_by(|a, b| b.1.cmp(&a.1));

    let lead_bound = if encoding == EUC_KR {
        0xC9
    } else if encoding == BIG5 {
        0xC7
    } else if encoding == GBK {
        0xD8
    } else if encoding == EUC_JP {
        0xD0
    } else {
        unreachable!();
    };
    let mut ret = Vec::new();

    let mut output = [0u8; 4];
    let mut input = [0u16; 1];
    let mut encoder = encoding.new_encoder();
    // We take 128 most frequent code points that encode
    // to the EUC square part of Level 1 / Hangul.
    // Specifically, for Big5, we don't want the part of
    // Level 1 Hanzi that does not overlap EUC.
    for (point, _) in sortable {
        let u = point as u16;
        input[0] = u;
        let (result, read, written) =
            encoder.encode_from_utf16_without_replacement(&input, &mut output, false);
        assert_eq!(result, EncoderResult::InputEmpty);
        assert_eq!(read, 1);
        if written != 2 {
            continue;
        }
        match (output[0], output[1]) {
            (0xA1..=0xFE, 0xA1..=0xFE) => {
                assert!(output[0] < lead_bound);
                ret.push(u);
                if ret.len() == 128 {
                    break;
                }
            }
            _ => {}
        }
    }

    ret
}

fn encoding_name_to_snake(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let under = lower.replace('-', "_");
    under
}

fn encoding_name_to_constant(name: &str) -> String {
    let lower = name.to_ascii_uppercase();
    let under = lower.replace('-', "_");
    under
}

fn generate_ascii_table(
    char_classes: &'static [&'static [char]],
    windows_encoding: &'static Encoding,
) -> Vec<u8> {
    let mut vec = Vec::new();
    vec.resize(128, 0u8);
    for (i, chars) in char_classes.iter().enumerate() {
        let class = i as u8;
        for &c in chars.iter() {
            if (c as usize) < vec.len() {
                vec[c as usize] = class;
            }

            let upper = if windows_encoding == WINDOWS_1254 && c == 'i' {
                'İ'
            } else if c == 'ς' {
                // Intentionally not handling final sigma to match
                // detection-time mapping.
                continue;
            } else if !c.is_lowercase() {
                // caseless
                continue;
            } else {
                let mut iter = c.to_uppercase();
                let first = iter.next().unwrap();
                if let Some(_) = iter.next() {
                    continue;
                }
                first
            };
            if (upper as usize) < vec.len() {
                vec[upper as usize] = class | 0x80;
            }
        }
    }
    vec
}

fn generate_upper_table(
    char_classes: &'static [&'static [char]],
    encoding: &'static Encoding,
) -> Vec<u8> {
    let mut vec = Vec::new();
    vec.resize(128, 0u8);
    let mut decoder = encoding.new_decoder_without_bom_handling();
    'outer: for i in 0..128 {
        let src = [(i as u8) + 128];
        let mut dst = [0u16];
        let (result, _, _) = decoder.decode_to_utf16_without_replacement(&src, &mut dst, false);
        match result {
            DecoderResult::Malformed(_, _) => {
                vec[i] = 255;
            }
            DecoderResult::InputEmpty => {
                let u = std::char::from_u32(dst[0] as u32).unwrap();
                if u >= '\u{80}' && u <= '\u{9F}' {
                    vec[i] = 255;
                    continue 'outer;
                }
                // if encoding == WINDOWS_1256 {
                //     if let Some(c) = ARABIC_FRENCH.iter().find(|&&x| x == u) {
                //         if c.is_uppercase() {
                //             vec[i] = 0xFE;
                //         } else {
                //             vec[i] = 0x7E;
                //         }
                //         continue 'outer;
                //     }
                // }
                for (j, chars) in char_classes.iter().enumerate() {
                    let class = j as u8;
                    for &c in chars.iter() {
                        if c == u {
                            vec[i] = class;
                            continue 'outer;
                        }
                        let upper = if encoding == WINDOWS_1254 && c == 'i' {
                            'İ'
                        } else if c == 'ς' {
                            // Intentionally not handling final sigma to match
                            // detection-time mapping.
                            continue;
                        } else if !c.is_lowercase() {
                            // caseless
                            continue;
                        } else {
                            let mut iter = c.to_uppercase();
                            let first = iter.next().unwrap();
                            if let Some(_) = iter.next() {
                                continue;
                            }
                            first
                        };
                        if upper == u {
                            vec[i] = class | 0x80;
                            continue 'outer;
                        }
                    }
                }
            }
            DecoderResult::OutputFull => {
                unreachable!();
            }
        }
    }

    vec
}

fn write_rs_file(
    rs: &Path,
    scores: &Vec<(Vec<u8>, &'static EncodingClass)>,
    cjk_scores: &Vec<(Vec<u16>, &'static str)>,
) {
    let file = File::create(rs).expect("Unable to create output file.");
    let mut writer = BufWriter::new(file);
    writer
        .write_all(b"/* Any copyright is dedicated to the Public Domain.\n")
        .unwrap();
    writer
        .write_all(b" * https://creativecommons.org/publicdomain/zero/1.0/ */\n\n")
        .unwrap();

    writer
        .write_all(
            b"use encoding_rs::Encoding;
use encoding_rs::WINDOWS_1258_INIT;
use encoding_rs::WINDOWS_1250_INIT;
use encoding_rs::ISO_8859_2_INIT;
use encoding_rs::WINDOWS_1251_INIT;
use encoding_rs::KOI8_U_INIT;
use encoding_rs::ISO_8859_5_INIT;
use encoding_rs::IBM866_INIT;
use encoding_rs::WINDOWS_1252_INIT;
use encoding_rs::WINDOWS_1253_INIT;
use encoding_rs::ISO_8859_7_INIT;
use encoding_rs::WINDOWS_1254_INIT;
use encoding_rs::WINDOWS_1255_INIT;
use encoding_rs::ISO_8859_8_INIT;
use encoding_rs::WINDOWS_1256_INIT;
use encoding_rs::ISO_8859_6_INIT;
use encoding_rs::WINDOWS_1257_INIT;
use encoding_rs::ISO_8859_13_INIT;
use encoding_rs::ISO_8859_4_INIT;
use encoding_rs::WINDOWS_874_INIT;
use super::IMPLAUSIBILITY_PENALTY;

",
        )
        .unwrap();

    writer
        .write_all(b"#[repr(align(64))] // Align to cache lines\n")
        .unwrap();
    writer.write_all(b"pub struct DetectorData {\n").unwrap();
    writer
        .write_all(b"    pub frequent_simplified: [u16; 128],\n")
        .unwrap();
    // writer
    //     .write_all(b"    pub frequent_traditional: [u16; 128],\n")
    //     .unwrap();
    writer
        .write_all(b"    pub frequent_kanji: [u16; 128],\n")
        .unwrap();
    writer
        .write_all(b"    pub frequent_hangul: [u16; 128],\n")
        .unwrap();

    writer.write_all(b"    latin_ascii: [u8; 128],\n").unwrap();
    writer
        .write_all(b"    non_latin_ascii: [u8; 128],\n")
        .unwrap();
    writer
        .write_all(b"    turkish_ascii: [u8; 128],\n")
        .unwrap();
    for encoding_class in ENCODING_CLASSES.iter() {
        for encoding in encoding_class.encodings {
            writer.write_all(b"    ").unwrap();
            writer
                .write_all(encoding_name_to_snake(encoding.name()).as_bytes())
                .unwrap();
            if encoding_class.name == "icelandic" {
                writer.write_all(b"_icelandic").unwrap();
            }
            writer.write_all(b": [u8; 128],\n").unwrap();
        }
    }
    for (vec, encoding_class) in scores {
        writer.write_all(b"    ").unwrap();
        writer.write_all(encoding_class.name.as_bytes()).unwrap();
        writer
            .write_fmt(format_args!(": [u8; {}],\n", vec.len()))
            .unwrap();
    }
    writer.write_all(b"}\n\n").unwrap();

    writer.write_all(b"#[rustfmt::skip]\n").unwrap();
    writer
        .write_all(b"pub static DETECTOR_DATA: DetectorData = DetectorData {\n")
        .unwrap();

    // ---

    for (scores, name) in cjk_scores.iter() {
        writer.write_all(b"    frequent_").unwrap();
        writer.write_all(name.as_bytes()).unwrap();
        writer.write_all(b": [\n").unwrap();
        write_cjk_frequency_table(&mut writer, scores);
        writer.write_all(b"    ],\n").unwrap();
    }

    // ---

    writer.write_all(b"    latin_ascii: [\n").unwrap();

    let western = encoding_class_by_encoding(WINDOWS_1252);
    write_class_mapping_table(
        &mut writer,
        &generate_ascii_table(western.char_classes, western.encodings[0]),
    );

    writer.write_all(b"    ],\n").unwrap();

    // ---

    writer.write_all(b"    non_latin_ascii: [\n").unwrap();

    let cyrillic = encoding_class_by_encoding(WINDOWS_1251);
    let cyrillic_ascii = generate_ascii_table(cyrillic.char_classes, cyrillic.encodings[0]);
    write_class_mapping_table(&mut writer, &cyrillic_ascii);

    writer.write_all(b"    ],\n").unwrap();

    // ---

    writer.write_all(b"    turkish_ascii: [\n").unwrap();

    let turkish = encoding_class_by_encoding(WINDOWS_1254);
    write_class_mapping_table(
        &mut writer,
        &generate_ascii_table(turkish.char_classes, turkish.encodings[0]),
    );

    writer.write_all(b"    ],\n").unwrap();

    // ---

    for encoding_class in ENCODING_CLASSES.iter() {
        for encoding in encoding_class.encodings {
            writer.write_all(b"    ").unwrap();
            writer
                .write_all(encoding_name_to_snake(encoding.name()).as_bytes())
                .unwrap();
            if encoding_class.name == "icelandic" {
                writer.write_all(b"_icelandic").unwrap();
            }
            writer.write_all(b": [\n").unwrap();

            let mut upper_table = generate_upper_table(encoding_class.char_classes, encoding);
            if *encoding == WINDOWS_1253 || *encoding == WINDOWS_1257 {
                // The corresponding ISO encodings map curly quotes. They
                // are classified as non-space-like to avoid misdetection.
                // Classify them as space-like for the Windows encoding to
                // avoid introducing differences between other Windows
                // encodings.
                upper_table[0x11] = 0;
                upper_table[0x12] = 0;
                upper_table[0x13] = 0;
                upper_table[0x14] = 0;
                upper_table[0x04] = 0;
            } else if *encoding == IBM866 {
                // Override class for no-break-space
                upper_table[127] = 3;
            }
            write_class_mapping_table(&mut writer, &upper_table);

            writer.write_all(b"    ],\n").unwrap();
        }
    }

    // ---

    for (vec, encoding_class) in scores {
        writer.write_all(b"    ").unwrap();
        writer.write_all(encoding_class.name.as_bytes()).unwrap();
        writer.write_all(b": [\n").unwrap();

        let (ascii_classes, non_ascii_classes) = count_ascii_classes(encoding_class.char_classes);
        write_probability_table(
            &mut writer,
            vec,
            ascii_classes,
            non_ascii_classes,
            encoding_class.char_classes,
        );

        writer.write_all(b"    ],\n").unwrap();
    }

    // ---

    writer.write_all(b"};\n\n").unwrap();

    // ---

    for encoding_class in ENCODING_CLASSES.iter() {
        let (ascii_classes, non_ascii_classes) = count_ascii_classes(encoding_class.char_classes);
        let upper = encoding_class.name.to_uppercase();
        writer
            .write_fmt(format_args!(
                "const {}_ASCII: usize = {};\n",
                upper, ascii_classes
            ))
            .unwrap();
        writer
            .write_fmt(format_args!(
                "const {}_NON_ASCII: usize = {};\n",
                upper, non_ascii_classes
            ))
            .unwrap();
    }

    // ===

    writer
        .write_all(
            b"#[inline(always)]
fn compute_index(
    x: usize,
    y: usize,
    ascii_classes: usize,
    non_ascii_classes: usize,
) -> Option<usize> {
    if x == 0 && y == 0 {
        return None;
    }
    if x < ascii_classes && y < ascii_classes {
        return None;
    }
    if y >= ascii_classes {
        return Some(
            (ascii_classes * non_ascii_classes)
                + (ascii_classes + non_ascii_classes) * (y - ascii_classes)
                + x,
        );
    }
    Some(y * non_ascii_classes + x - ascii_classes)
}

pub struct SingleByteData {
    pub encoding: &'static Encoding,
    lower: &'static [u8; 128],
    upper: &'static [u8; 128],
    probabilities: &'static [u8],
    ascii: usize,
    non_ascii: usize,
    letter_boundary: u8,
}

impl SingleByteData {
    #[inline(always)]
    pub fn classify(&'static self, byte: u8) -> u8 {
        let high = byte >> 7;
        let low = byte & 0x7F;
        if high == 0u8 {
            self.lower[usize::from(low)]
        } else {
            self.upper[usize::from(low)]
        }
    }

    #[inline(always)]
    pub fn is_latin_alphabetic(&'static self, byte: u8) -> bool {
        return byte >= 2 && byte < self.letter_boundary;
    }

    #[inline(always)]
    pub fn is_non_latin_alphabetic(&'static self, byte: u8) -> bool {
        return byte >= self.letter_boundary;
    }

    #[inline(always)]
    pub fn score(&'static self, current_class: u8, previous_class: u8) -> i64 {
        if let Some(index) = compute_index(
            usize::from(previous_class),
            usize::from(current_class),
            self.ascii,
            self.non_ascii,
        ) {
            let b = self.probabilities[index];
            if b == 255 {
                IMPLAUSIBILITY_PENALTY
            } else {
                i64::from(b)
            }
        } else {
            0
        }
    }
}

impl PartialEq for SingleByteData {
    #[inline]
    fn eq(&self, other: &SingleByteData) -> bool {
        (self as *const SingleByteData) == (other as *const SingleByteData)
    }
}

",
        )
        .unwrap();

    writer
        .write_all(b"pub static SINGLE_BYTE_DATA: [SingleByteData; 20] = [\n")
        .unwrap();

    for encoding_class in ENCODING_CLASSES.iter() {
        let class_upper = encoding_class.name.to_ascii_uppercase();
        for encoding in encoding_class.encodings {
            let windows_encoding = encoding_class.encodings[0];
            let lower = if windows_encoding == WINDOWS_1254 {
                "turkish_ascii"
            } else if is_latin(windows_encoding) {
                "latin_ascii"
            } else {
                "non_latin_ascii"
            };
            let encoding_upper = encoding_name_to_constant(encoding.name());
            let mut encoding_snake = encoding_name_to_snake(encoding.name());
            if encoding_class.name == "icelandic" {
                encoding_snake.push_str("_icelandic");
            }
            let letter_boundary = find_letter_boundary(encoding_class);
            writer
                .write_fmt(format_args!(
                    "    SingleByteData {{
        encoding: &{}_INIT,
        lower: &DETECTOR_DATA.{},
        upper: &DETECTOR_DATA.{},
        probabilities: &DETECTOR_DATA.{},
        ascii: {}_ASCII,
        non_ascii: {}_NON_ASCII,
        letter_boundary: {},
    }},\n",
                    encoding_upper,
                    lower,
                    encoding_snake,
                    encoding_class.name,
                    class_upper,
                    class_upper,
                    letter_boundary,
                ))
                .unwrap();
        }
    }

    writer.write_all(b"];\n\n").unwrap();

    let mut i = 0;
    for encoding_class in ENCODING_CLASSES.iter() {
        for encoding in encoding_class.encodings {
            let mut upper = encoding_name_to_constant(encoding.name());
            if encoding_class.name == "icelandic" {
                upper.push_str("_ICELANDIC");
            }
            writer
                .write_fmt(format_args!("pub const {}_INDEX: usize = {};\n", upper, i))
                .unwrap();
            i += 1;
        }
    }
}

fn write_byte(writer: &mut dyn Write, byte: u8) {
    if byte < 10 {
        writer.write_fmt(format_args!("  {},", byte)).unwrap();
    } else if byte < 100 {
        writer.write_fmt(format_args!(" {},", byte)).unwrap();
    } else {
        writer.write_fmt(format_args!("{},", byte)).unwrap();
    }
}

fn write_class_mapping_table(writer: &mut dyn Write, table: &[u8]) {
    assert_eq!(table.len(), 128);
    for i in 0..8 {
        writer.write_all(b"        ").unwrap();
        for j in 0..16 {
            let index = i * 16 + j;
            write_byte(writer, table[index]);
        }
        writer.write_all(b"\n").unwrap();
    }
}

fn write_cjk_frequency_table(writer: &mut dyn Write, table: &[u16]) {
    assert_eq!(table.len(), 128);
    for i in 0..8 {
        writer.write_all(b"        ").unwrap();
        for j in 0..16 {
            let index = i * 16 + j;
            writer
                .write_fmt(format_args!("0x{:X}, ", table[index]))
                .unwrap();
        }
        writer.write_all(b"\n").unwrap();
    }
}

fn write_probability_table(
    writer: &mut dyn Write,
    table: &[u8],
    ascii: usize,
    non_ascii: usize,
    char_classes: &'static [&'static [char]],
) {
    let side = ascii + non_ascii;
    for row in 0..side {
        writer.write_all(b"        ").unwrap();
        for column in 0..side {
            if let Some(index) = compute_index(column, row, ascii, non_ascii) {
                // writer.write_fmt(format_args!("/*{}*/", index)).unwrap();
                write_byte(writer, table[index]);
            } else {
                writer.write(b"    ").unwrap();
            }
        }
        writer.write_all(b" // ").unwrap();
        writer
            .write_fmt(format_args!("{},", char_classes[row][0]))
            .unwrap();
        writer.write_all(b"\n").unwrap();
    }
    writer.write_all(b"      //").unwrap();
    for column in 0..side {
        writer
            .write_fmt(format_args!("  {},", char_classes[column][0]))
            .unwrap();
    }
    writer.write_all(b"\n").unwrap();
}

fn encoding_class_by_encoding(encoding: &'static Encoding) -> &EncodingClass {
    for encoding_class in ENCODING_CLASSES.iter() {
        for enc in encoding_class.encodings.iter() {
            if enc == &encoding {
                return encoding_class;
            }
        }
    }
    unreachable!();
}

fn download_corpus(dir: &Path) {
    let prefix = "https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/";
    let date = "20190420";
    let mut curl = Command::new("curl");
    curl.current_dir(dir);
    curl.arg("-L");
    curl.arg("--remote-name-all");
    for encoding_class in ENCODING_CLASSES.iter() {
        for lang in encoding_class.languages.iter() {
            let mut url = String::new();
            url.push_str(prefix);
            url.push_str(lang);
            url.push_str("wiki/");
            url.push_str(date);
            url.push_str("/");
            url.push_str(lang);
            url.push_str("wiki-");
            url.push_str(date);
            url.push_str("-pages-articles.xml.bz2");
            curl.arg(url);
        }
    }
    curl.output().expect("Executing curl failed");
}

fn count_pairs(dir: &Path) {
    ENCODING_CLASSES.par_iter().for_each(|c| c.count(dir));
}

fn cjk_bounds(lang: &str) -> (usize, usize) {
    if lang == "ko" {
        (0xAC00, 0xD7A4)
    } else {
        (0x4E00, 0x9FA6)
    }
}

fn count_cjk(dir: &Path, lang: &str, start: usize, end: usize) {
    eprintln!("Counting {:?}", lang);
    let mut scores = Vec::new();
    scores.resize(end - start, 0u64);
    let corpus = find_file(dir, lang);
    let iter = open_bzip2(&corpus);
    for c in iter {
        let offset = (c as usize).wrapping_sub(start);
        if offset < scores.len() {
            scores[offset] += 1;
        }
    }
    let path = gen_count_path(dir, lang, false);
    write_scores(&path, &scores);
}

fn main() {
    let mut args = std::env::args_os();
    if args.next().is_none() {
        eprintln!("Error: Program name missing from arguments.");
        std::process::exit(-1);
    }
    if let Some(command) = args.next() {
        if "download" == command {
            if let Some(path) = args.next() {
                download_corpus(Path::new(&path));
            } else {
                eprintln!("Error: Download directory missing.");
                std::process::exit(-3);
            }
        } else if "count" == command {
            if let Some(path) = args.next() {
                let dir = Path::new(&path);
                rayon::join(
                    || count_pairs(dir),
                    || {
                        CJK_ENCODINGS.par_iter().for_each(|(lang, _, _)| {
                            let (start, end) = cjk_bounds(lang);
                            count_cjk(dir, lang, start, end)
                        })
                    },
                );
            } else {
                eprintln!("Error: Download directory missing.");
                std::process::exit(-3);
            }
        } else if "generate" == command {
            if let Some(dir) = args.next() {
                if let Some(rs) = args.next() {
                    train_with_dir(Path::new(&dir), Path::new(&rs));
                } else {
                    eprintln!("Error: Target Rust file missing.");
                    std::process::exit(-3);
                }
            } else {
                eprintln!("Error: Download directory missing.");
                std::process::exit(-3);
            }
        } else {
            eprintln!("Error: Unknown command.");
            std::process::exit(-3);
        }
    } else {
        eprintln!("Error: Command missing.");
        std::process::exit(-2);
    };
}

#[cfg(test)]
mod tests {
    use super::compute_index;
    #[test]
    fn test_compute_index() {
        //    0  1  2  3  4
        //  +--+--+--+--+--+
        // 0|  |  | 0| 1| 2|
        //  +--+--+--+--+--+
        // 1|  |  | 3| 4| 5|
        //  +--+--+--+--+--+
        // 2| 6| 7| 8| 9|10|
        //  +--+--+--+--+--+
        // 3|11|12|13|14|15|
        //  +--+--+--+--+--+
        // 4|16|17|18|19|20|
        //  +--+--+--+--+--+
        assert_eq!(compute_index(1, 1, 2, 3), None);
        assert_eq!(compute_index(2, 0, 2, 3), Some(0));
        assert_eq!(compute_index(4, 0, 2, 3), Some(2));
        assert_eq!(compute_index(2, 1, 2, 3), Some(3));
        assert_eq!(compute_index(4, 1, 2, 3), Some(5));
        assert_eq!(compute_index(0, 2, 2, 3), Some(6));
        assert_eq!(compute_index(4, 2, 2, 3), Some(10));
        assert_eq!(compute_index(0, 3, 2, 3), Some(11));
        assert_eq!(compute_index(4, 3, 2, 3), Some(15));
        assert_eq!(compute_index(0, 4, 2, 3), Some(16));
        assert_eq!(compute_index(4, 4, 2, 3), Some(20));
    }
}
