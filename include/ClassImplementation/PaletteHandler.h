//
// Created by progamers on 5/6/25.
//
#pragma once
#include "HardCodedVars.h"
#include "FractalClass.cuh"
#include <iostream>
#include "Macros.h"
template <typename Derived>
void FractalBase<Derived>::setPallete(std::string name) {
    auto it = s_palette_name_to_enum.find(name);

    if (it != s_palette_name_to_enum.end()) {
        Palletes target_palette_enum = it->second;

        const unsigned int palette_size = BASIC_PALETTE_SIZE;
        bool creation_successful = true;

        switch (target_palette_enum) {
            case Palletes::HSV:
                // Call the specific HSV function which needs the extra argument
                palette = createHSVPalette(palette_size, degrees_offsetForHSV);
                break;
            case Palletes::Basic:
                palette = BluePlusBlackWhitePalette(palette_size);
                break;
            case Palletes::BlackOWhite:
                palette = CreateBlackOWhitePalette(palette_size);
                break;
            case Palletes::OscillatingGrayscale:
                palette = CreateOscillatingGrayscalePalette(palette_size);
                break;
            case Palletes::Interpolated:
                palette = CreateInterpolatedPalette(palette_size);
                break;
            case Palletes::Pastel:
                palette = CreatePastelPalette(palette_size);
                break;
            case Palletes::CyclicHSV:
                palette = CreateCyclicHSVPpalette(palette_size);
                break;
            case Palletes::Fire:
                palette = CreateFirePalette(palette_size);
                break;
            case Palletes::FractalPattern:
                palette = CreateFractalPatternPalette(palette_size);
                break;
            case Palletes::PerlinNoise:
                palette = CreatePerlinNoisePalette(palette_size);
                break;
            case Palletes::Water:
                palette = CreateWaterPalette(palette_size);
                break;
            case Palletes::Sunset:
                palette = CreateSunsetPalette(palette_size);
                break;
            case Palletes::DeepSpace:
                palette = CreateDeepSpaceWideVeinsPalette(palette_size);
                break;
            case Palletes::Physchodelic:
                palette = CreatePsychedelicWavePalette(palette_size);
                break;
            case Palletes::IceCave:
                palette = CreateIceCavePalette(palette_size);
                break;
            case Palletes::AccretionDisk:
                palette = CreateAccretionDiskPalette(palette_size);
                break;
            case Palletes::ElectricNebula:
                palette = CreateElectricNebulaPalette(palette_size);
                break;
            case Palletes::Random:
                palette = CreateRandomPalette(palette_size);
                break;

            default:
                std::cerr << "Internal error: Unhandled palette enum value ("
                          << static_cast<int>(target_palette_enum) << ") in setPallete." << std::endl;
                creation_successful = false;
                break;
        }

        if (creation_successful) {
            paletteSize = palette_size;
            curr_pallete = target_palette_enum;
            if(isCudaAvailable) {
                COPY_PALETTE_TO_DEVICE(palette.data(), d_palette, cu_palette, context);
            }
        }

    } else {
        std::cerr << "Warning: Unknown palette name requested: \"" << name << "\". Palette not changed." << std::endl;
    }
}

template <typename Derived>
void FractalBase<Derived>::SetDegreesOffsetForHSV(int degrees) { degrees_offsetForHSV = degrees; setPallete("HSV"); }